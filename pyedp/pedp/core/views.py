#  Copyright 2021 Intel Corporation
#  This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
#  express license under which they were provided to you (License). Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
#  prior written permission.
#
#  This software and the related documents are provided as is, with no express or implied warranties, other than those
#  that are expressly stated in the License.
#
#
from dataclasses import dataclass
from enum import Enum, auto, Flag
from typing import List, Callable, Union, Dict, Type
from abc import abstractmethod

import numpy as np
import pandas as pd
from tdigest import tdigest

from pedp.core.metric_computer import MetricComputer
from pedp.core.normalizer import Normalizer
from pedp.core.types import RawEmonDataFrameColumns as redc, RawEmonDataFrame, SummaryViewDataFrameColumns as svdc, \
    MetricDefinition, EventInfoDataFrame, EventInfoDataFrameColumns as eidc, Device
from pedp.core.internals.types import StatisticsDataFrame, StatisticsDataFrameColumns as sdf

RenameColumnsCallable = Union[None, Callable[[pd.DataFrame], List[str]]]


class ViewType(Flag):
    """
    Supported view types. Can be combined using bitwise OR, e.g. SUMMARY | DETAILS
    """
    SUMMARY = auto()
    DETAILS = auto()
    ALL = SUMMARY | DETAILS


class ViewAggregationLevel(Enum):
    """
    Supported view aggregation levels
    """
    SYSTEM = auto()
    SOCKET = auto()
    CORE = auto()
    THREAD = auto()
    UNCORE = auto()


@dataclass(frozen=True)
class ViewAttributes:
    """
    View attributes
    """
    view_name: str  # a name that uniquely identifies this view
    view_type: ViewType  # view type, e.g. Summary, Details, ...
    aggregation_level: ViewAggregationLevel  # data aggregation level, e.g. System, Socket, Core, ...
    device: Union[Device, None]  # device for this view # TODO: this may be required (can't be None)
    show_modules: bool  # show module information for this view
    metric_computer: Union[MetricComputer, None]  # metric computer assigned to the view
    normalizer: Union[Normalizer, None]  # normalizer assigned to the view
    required_events: Union[EventInfoDataFrame, None]  # column names that must appear in the view data

    def clone(self, update: Dict = None):
        """
        Creates a copy of the ViewAttributes object

        :param update: a dict with updated attribute values to assign to the new object

        :return: a new copy of the object
        """
        new_attr = self.__dict__.copy()
        if update:
            new_attr.update(update)
        return ViewAttributes(**new_attr)


@dataclass
class ViewData:
    attributes: ViewAttributes
    data: pd.DataFrame


class ViewCollection:
    """
    Stores information about the EDP views to generate
    """

    def __init__(self):
        """
        Initializes an empty view collection.
        """
        self.__view_configurations: List[ViewAttributes] = []

    @property
    def views(self) -> List[ViewAttributes]:
        return self.__view_configurations.copy()

    def append_views(self, views: List[ViewAttributes]) -> None:
        """
        Appends a list of ViewAttributes to the internal __view_configurations

        @param: views, A list of ViewAttributes to be appended
        @return: None, updates the internal __view_configuration list of ViewAttributes
        """
        self.__view_configurations.extend(views)

    def add_view(self,
                 view_name: str,
                 view_type: ViewType,
                 aggregation_level: ViewAggregationLevel,
                 device: Device,
                 show_modules: bool,
                 metric_computer: MetricComputer = None,
                 normalizer: Normalizer = None,
                 required_events: EventInfoDataFrame = None) -> object:
        """
        Add a view configuration

        :param view_name: a name that uniquely identify this view
        :param view_type: the view type to add, e.g. Summary, Details. You can specify multiple types
                          (e.g. Summary and Details) by combining types using bitwise OR.
        :param aggregation_level: the level at which to aggregate data, e.g. System, Socket, Core...
        @param device: device to be filtered for this view
        :param show_modules: include module information if True
        :param metric_computer: the metric computer to use for the specified view
        :param normalizer: the normalizer to use for the specified view
        :param required_events: Events required to generate the specified view
        """

        def validate_preconditions():
            if view_name in [config.view_name for config in self.__view_configurations]:
                raise ValueError(f'a view with the name "{view_name}" already exists. '
                                 f'Duplicate view names not allowed')

        validate_preconditions()

        if device is not None and required_events is not None:
            required_events = self._filter_core_type_events(required_events, device)

        view_attr = ViewAttributes(view_name, view_type, aggregation_level, device, show_modules, metric_computer,
                                   normalizer, required_events)
        self.__view_configurations.append(view_attr)

    @staticmethod
    def _filter_core_type_events(event_info: EventInfoDataFrame, device: Device) -> EventInfoDataFrame:
        # TODO: change function name? see if this applies to all devices
        if (eidc.DEVICE) in event_info.columns:
            event_info = event_info.loc[~(event_info[eidc.DEVICE].isin(device.exclusions))]
        return event_info


class ViewGenerator:
    """
    Generate views with various levels of data aggregation (per system, socket, core, thread...)
    """

    def __init__(self, view_collection: ViewCollection = None):
        """
        Initialize view generator

        :param view_collection: views to generate
        """

        def validate_preconditions():
            if view_collection is None or len(view_collection.views) == 0:
                raise ValueError('at least one view is required but none provided')

        validate_preconditions()
        self.__views: List[_DataView] = []
        self.__initialize_view_definitions(view_collection)

    @property
    def views(self):
        return self.__views

    def generate_detail_views(self, df: RawEmonDataFrame) -> Dict[str, ViewData]:
        """
        Process the input dataframe and generate data for all Detail Views

        :param df: input data frame

        :return: A list of `ViewData` objects, one for each Details View specified in the view generator configuration.
        """
        results = {}
        if df.empty:
            return results

        for view in filter(lambda v: ViewType.DETAILS in v.attributes.view_type, self.__views):
            details_view = view.generate_details(df)
            if details_view is not None and not details_view.data.empty:
                results[view.attributes.view_name] = details_view

        return results

    def compute_aggregates(self, df: RawEmonDataFrame) -> List[ViewData]:
        """
        Computes aggregated sums of event values for a raw input dataframe and each view type specified in the view
        generator configuration

        :@param df: RawEmonDataFrame input

        :return: aggregated event values for each view type specified in the view generator configuration
        """
        if df.empty:
            return [ViewData(view.attributes, df) for view in self.__views if ViewType.SUMMARY in
                    view.attributes.view_type]
        return [view.compute_aggregate(df) for view in self.__views if ViewType.SUMMARY in view.attributes.view_type]

    def __initialize_view_definitions(self, view_collection: ViewCollection):
        def create_view_definition(config: ViewAttributes):
            return {
                ViewAggregationLevel.SYSTEM: _SystemDataView,
                ViewAggregationLevel.SOCKET: _SocketDataView,
                ViewAggregationLevel.CORE: _CoreDataView,
                ViewAggregationLevel.THREAD: _ThreadDataView,
                ViewAggregationLevel.UNCORE: _UncoreDataView,
            }[config.aggregation_level](config)

        self.__views = list(map(lambda config: create_view_definition(config), view_collection.views))


class _BaseStat:

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute aggregated statistic/s for an input dataframe
        """
        pass

    @abstractmethod
    def get_stats_values(self) -> pd.DataFrame:
        """
        Return the resulting dataframe after computing the statistic/s
        """
        pass


class _MinMax(_BaseStat):

    def __init__(self):
        self.__stats_df = None
        self.__columns = [sdf.MIN, sdf.MAX]

    def compute(self, df: pd.DataFrame) -> None:
        stats = [np.min, np.max]
        block_stats_df = df.agg(stats, axis='index', skipna=True, numeric_only=True).T
        block_stats_df.columns = self.__columns
        if self.__stats_df is None:
            self.__stats_df = block_stats_df.copy()
        else:
            self.__stats_df[sdf.MIN] = np.fmin(self.__stats_df[sdf.MIN], block_stats_df[sdf.MIN])
            self.__stats_df[sdf.MAX] = np.fmax(self.__stats_df[sdf.MAX], block_stats_df[sdf.MAX])

    def get_stats_values(self) -> pd.DataFrame:
        return self.__stats_df


class _Percentile(_BaseStat):

    def __init__(self, percentile=95):
        self.__event_percentiles: Dict[str, tdigest.TDigest] = {}
        self.__percentile = percentile
        self.__columns = sdf.PERCENTILE

    def compute(self, df: pd.DataFrame) -> None:
        for column in df:
            if column not in self.__event_percentiles:
                self.__event_percentiles[column] = tdigest.TDigest()
            values = df[column].values
            values = values[np.isfinite(values)]  # remove nans and infinite values
            self.__event_percentiles[column].batch_update(values)

    def get_stats_values(self) -> pd.DataFrame:
        percentile_df = pd.DataFrame([[item[0], item[1].percentile(self.__percentile)]
                                      for item in self.__event_percentiles.items()
                                      if item[1].C.count > 0],
                                     columns=['name', self.__columns]).set_index('name')
        return percentile_df


class _Variation(_BaseStat):

    def __init__(self):
        self.__event_variance = {}
        self.__event_aggregate = {}
        self.__tmp_aggregate = (0, 0, 0)
        self.__columns = [sdf.VARIATION]

    def compute(self, df: pd.DataFrame) -> None:
        # Implementation of Welford's Online Algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        for column in df:
            self.__tmp_aggregate = self.__event_aggregate.get(column, (0, 0, 0))
            values = df[column].dropna()
            if len(values) > 0:
                self.__event_variance[column] = values.apply(self.__compute_variance_value).iloc[-1]
            self.__event_aggregate[column] = self.__tmp_aggregate

    def get_stats_values(self) -> pd.DataFrame:
        stats_values_dict = dict(zip(self.__columns, [self.__event_variance]))
        stats_values_df = pd.DataFrame(stats_values_dict)
        return stats_values_df

    @staticmethod
    def __update_aggregate_value(existing_aggregate, new_value):
        (count, mean, M2) = existing_aggregate
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        M2 += delta * delta2
        return count, mean, M2

    @staticmethod
    def __get_variation(existing_aggregate):
        # Retrieve the mean, variance and sample variance from an aggregate
        (count, mean, M2) = existing_aggregate
        if count < 2:
            return 0  # should be null, but ruby outputs this as 0
        else:
            variance_over_mean = (np.sqrt(M2 / count)) / mean
        return variance_over_mean

    def __compute_variance_value(self, value):
        self.__tmp_aggregate = self.__update_aggregate_value(self.__tmp_aggregate, value)
        return self.__get_variation(self.__tmp_aggregate)


class _Statistics:
    """
    Compute various statistics (min, max, percentile) for events and metrics
    """

    def __init__(self):
        self.__base_stats = [_MinMax(), _Percentile(95), _Variation()]
        self.__stats_df: Union[StatisticsDataFrame, None] = None

    def compute(self, df: pd.DataFrame) -> None:
        """
        Compute statistics from the input data frame and update object state

        :param df: input data frame
        """
        if df.empty:
            return
        for stat in self.__base_stats:
            stat.compute(df)

    def get_statistics(self) -> StatisticsDataFrame:
        """
        :return: a data frame with the computed events and metrics statistics (min, max, percentile...)
        """
        stat_data_frames = [s.get_stats_values() for s in self.__base_stats]
        self.__stats_df = pd.concat(stat_data_frames, axis=1)
        return StatisticsDataFrame(self.__stats_df)


class _DataView:
    """
    Define the common parameters and logic for generating an EDP data view
    """

    def __init__(self,
                 config: ViewAttributes,
                 summary_group_by: List[str],
                 aggregator_columns: List[str],
                 aggregator_group_by: List[str],
                 details_group_by: List[str],
                 details_index: List[str],
                 device_filter=None,
                 device_filter_mode='include'
                 ):
        """
        Initialize view definition

        :param config: view attributes
        :param summary_group_by: columns to group by for summary view
        :param aggregator_columns: columns to use for computing aggregated values
        :param aggregator_group_by: columns to group by for computing aggregation (a subset of aggregator_columns)
        :param details_group_by: columns to group by for details view
        :param details_index: columns whose values will be used to determine the index of the details view
        :param device_filter: an optional device name (e.g., CORE, CHA...). When specified, only events related
                              to the specified device will be included in the generated view. This will also
                              limit the computed metrics to those that only use events related to the specified
                              device.
        """
        self.__summary_group_by = summary_group_by
        self._aggregator_columns = aggregator_columns
        self._aggregator_group_by = aggregator_group_by
        self.__details_group_by = details_group_by
        self.__details_index = details_index
        self.__device_filter = device_filter
        self.__device_filter_mode = device_filter_mode
        self.__normalizer = self.__get_config_value(config.normalizer, _NullNormalizer())
        self.__metric_computer = self.__get_config_value(config.metric_computer, _NullMetricComputer([]))
        self.__required_events = self.__get_config_value(config.required_events,
                                                         pd.DataFrame())
        self._attributes = config
        self.__event_summary_values = pd.DataFrame()

    @property
    def attributes(self) -> ViewAttributes:
        return self._attributes

    def update_summary(self, df: RawEmonDataFrame, event_summary_values: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """
        Process the input dataframe and update summary statistics

        @param df: input data frame of new event values
        @param event_summary_values: persisted summary values from DataAccumulator
        """
        if df.empty or ViewType.SUMMARY not in self.attributes.view_type:
            return df

        view_data = self._update_summary_values(df, event_summary_values)
        return view_data

    def update_statistics(self, details_view_df: Union[RawEmonDataFrame, pd.DataFrame]) -> pd.DataFrame:
        # A method that subclasses can override to update statistics in summary views
        return pd.DataFrame()

    def generate_details(self, df: RawEmonDataFrame) -> ViewData:
        """
        Process the input dataframe and return details data

        :param df: input data frame

        :return: a `DetailsData` object. The `data` member contain the details view data, or an empty data frame if the
                 view is not configured to generate details data
        """
        details_view_df = self._generate_details_dataframe(df) if ViewType.DETAILS in self.attributes.view_type \
            else pd.DataFrame()
        return ViewData(self.attributes.clone(update={'view_type': ViewType.DETAILS}), details_view_df)

    def compute_aggregate(self, df: RawEmonDataFrame) -> ViewData:
        """
        Return summary data

        :return: a dataframe with the summary view data
        """
        df = self.__filter_devices(df, redc)
        df_agg_values = df[self._aggregator_columns].groupby(self._aggregator_group_by, observed=True).sum()
        return ViewData(self.attributes.clone(update={'view_type': ViewType.SUMMARY}), df_agg_values)

    def generate_summary(self, event_df: pd.DataFrame) -> ViewData:
        """
        Return summary data

        :return: a dataframe with the summary view data
        """
        summary_view_df = self._generate_summary_dataframe(self.__normalizer.normalize(event_df,
                                                                                       event_axis='index')) \
            if ViewType.SUMMARY in self.attributes.view_type else pd.DataFrame()
        return ViewData(self.attributes.clone(update={'view_type': ViewType.SUMMARY}), summary_view_df)

    def _generate_summary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        summary_events_df = pd.DataFrame(df.groupby(self.__summary_group_by, observed=True).sum()[redc.VALUE])
        # Reshape the dataframe to have event names as columns
        if summary_events_df.index.nlevels == 1:
            summary_events_df = pd.DataFrame(summary_events_df).transpose()
        else:
            summary_events_df = summary_events_df.unstack(level=redc.NAME).droplevel(0, axis=1)
        summary_metrics_df = self._compute_metrics(summary_events_df)
        summary_view_df = pd.concat([summary_metrics_df, summary_events_df], axis=1)

        return summary_view_df

    def _compute_metrics(self, df: pd.DataFrame, calculate_block_level=False) -> pd.DataFrame:
        constant_values = self._override_constant_values(self.__metric_computer.symbol_table)
        metrics_df = self.__metric_computer.compute_metric(
            df, constant_values, calculate_block_level=calculate_block_level)
        return metrics_df

    def _generate_details_dataframe(self, df: RawEmonDataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        details_events_df = self.__filter_devices(df, redc)
        # TODO: add test where __filter_devices produces an empty dataframe (for uncore views?)
        details_events_df = self.__normalizer.normalize(details_events_df)
        details_events_df = details_events_df.groupby(
            self.__details_group_by, as_index=False, observed=True).sum(numeric_only=True).reset_index()
        details_view_df = pd.pivot_table(details_events_df,
                                         columns=[redc.NAME],
                                         index=self.__details_index,
                                         values=redc.VALUE)
        details_view_df = self.__adjust_details_view_dataframe_columns(details_view_df)
        details_metrics_df = self._compute_metrics(details_view_df, calculate_block_level=True)
        details_view_df = self.__merge_events_and_metrics_detail_views(details_view_df, details_metrics_df)
        details_view_df = details_view_df.reset_index(level=redc.GROUP, drop=True)
        return details_view_df

    def _update_summary_values(self, df: pd.DataFrame, event_summary_values: pd.DataFrame) -> pd.DataFrame:
        new_event_summary_values = df.reset_index()[self._aggregator_columns].groupby(self._aggregator_group_by,
                                                                                      observed=True).sum()
        if event_summary_values.empty:
            event_summary_values = new_event_summary_values.copy()
        elif len(set(df.columns).difference(set(event_summary_values.columns))) > 0:
            event_summary_values = event_summary_values.merge(new_event_summary_values, how='left', left_index=True,
                                                              right_index=True)
        else:
            event_summary_values = event_summary_values.add(new_event_summary_values, fill_value=0)
        return event_summary_values

    @staticmethod
    def __merge_events_and_metrics_detail_views(details_view_df: pd.DataFrame,
                                                details_metrics_df: pd.DataFrame) -> pd.DataFrame:
        if details_metrics_df.empty:
            return details_view_df

        details_view_df = pd.merge(details_metrics_df, details_view_df, left_index=True, right_index=True)
        return details_view_df

    def _override_constant_values(self, symbol_table: Dict) -> Dict:
        # A method that subclasses can override to modify the constant values to be used for computing metrics
        return symbol_table

    def __adjust_details_view_dataframe_columns(self, details_view_df: pd.DataFrame):
        if self.__required_events.empty:
            return details_view_df

        # Force details view columns and their order based on the `ViewGenerator` configuration
        required_events_df = self.__filter_devices(self.__required_events, eidc)
        return details_view_df.reindex(columns=required_events_df[eidc.NAME])

    @staticmethod
    def __get_config_value(value, alternate_value_if_none):
        return value if value is not None else alternate_value_if_none

    def __filter_devices(self, df: Union[pd.DataFrame, RawEmonDataFrame], df_column_type: Union[Type[redc],
                                                                                                Type[eidc]]):
        if self.__device_filter_mode not in ['include', 'exclude']:
            raise ValueError("device_filter_mode must be either 'include' or 'exclude'")
        if self.__device_filter and self.__device_filter_mode == 'include':
            df = df.loc[df[df_column_type.DEVICE].isin(self.__device_filter)]
        elif self.__device_filter and self.__device_filter_mode == 'exclude':
            df = df.loc[~(df[df_column_type.DEVICE].isin(self.__device_filter))]
        return df


class DataAccumulator:
    """
    Accumulates event counts for summary views with various levels of data aggregation (per system, socket, core,
    thread...)
    Also updates statistics for these views (currently only the system summary view)
    """

    def __init__(self, view_generator: ViewGenerator):
        def validate_preconditions():
            if len(view_generator.views) == 0:
                raise ValueError('view_collection must have at least one view')

        validate_preconditions()
        self.__views = {view.attributes.view_name: view for view in view_generator.views}
        self.__event_views = {view.attributes.view_name: ViewData(view.attributes, pd.DataFrame())
                              for view in view_generator.views}
        self.__stats_views = {view.attributes.view_name: ViewData(view.attributes, pd.DataFrame())
                              for view in view_generator.views}

    def get_event_summaries(self) -> Dict[str, ViewData]:
        return self.__event_views

    def get_statistics(self) -> Dict[str, ViewData]:
        return self.__stats_views

    def update_aggregates(self, summary_computations: List[ViewData]) -> None:
        for new_view in summary_computations:
            data_view = self.__views[new_view.attributes.view_name]
            view = self.__event_views[new_view.attributes.view_name]
            view.data = data_view.update_summary(new_view.data, view.data)

    def update_statistics(self, detail_views: Dict[str, ViewData] = None, df: RawEmonDataFrame = None) -> None:
        """
        Updates summary statistics for a list of detail views or a RawEmonDataFrame.
        Takes in either a list of detail views or a RawEmonDataFrame. Only pass in a RawEmonDataFrame if you do not
        wish to generate detail views, otherwise only pass in detail_views
        @param detail_views: a list of detail_views required to update summary statistics
        @param df: a raw Emon or perfmon dataframe
        @return: None, internal stats_views will be updated and persisted inside of DataAccumulator
        """

        def validate_preconditions():
            if detail_views is None and df is None:
                raise ValueError('if no detail views are requested, then a dataframe must be passed into this method')

        validate_preconditions()
        for view_id, view in self.__stats_views.items():
            if detail_views and view.attributes.view_type != ViewType.DETAILS:
                df = detail_views[view_id.replace('summary', 'details')].data
            if view.attributes.view_type == ViewType.DETAILS:
                continue
            data_view = self.__views[view_id]
            view.data = data_view.update_statistics(df)

    def generate_summary_views(self) -> Dict[str, ViewData]:
        """
        Computes metrics for each view and updates statistics when needed.
        :return: summary view for each view type specified in the view generator configuration
        """
        summary_data_views = dict()
        for idx, (key, view) in enumerate(self.__event_views.items()):
            summary_data_views.update({key: (view, self.__views[key])})
        summary_views = {}
        for view_id, (summary_view, data_view) in summary_data_views.items():
            stats_view = self.__stats_views[view_id]
            if ViewType.SUMMARY in summary_view.attributes.view_type:
                summary_view = data_view.generate_summary(summary_view.data)
                summary_view.data = pd.concat([summary_view.data, stats_view.data.transpose()])
                summary_views[summary_view.attributes.view_name] = summary_view
        return summary_views


class _SystemDataView(_DataView):
    """
    View definition for the System summary and detail views

    Overrides the behavior of _DataView

    The `update_statistics()` method always compute the data for the details view unless detail view data is provided.
    This is needed to compute event and metric statistics for the summary view.

    The `generate_summary()` method adds event and metric statistics to the summary dataframe.
    """

    def __init__(self, config: ViewAttributes):
        super().__init__(config,
                         summary_group_by=[redc.NAME],
                         aggregator_columns=[redc.NAME, redc.SOCKET, redc.UNIT, redc.TSC, redc.VALUE],
                         aggregator_group_by=[redc.NAME, redc.SOCKET, redc.UNIT],
                         details_group_by=[redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME],
                         details_index=[redc.GROUP, redc.TIMESTAMP],
                         device_filter=config.device.exclusions,
                         device_filter_mode='exclude'
                         )
        self.__stats = _Statistics()

    def update_statistics(self, df: Union[RawEmonDataFrame, pd.DataFrame]) -> StatisticsDataFrame:
        """
        Supports updating statistics when passed a raw EMON dataframe or an already generated details view.

        @param df: a raw EMON dataframe or details view dataframe
        @return: a summary dataframe with updated statistics
        """
        if self.__is_raw_emon_dataframe(df):
            details_view_df = self._generate_details_dataframe(df)
        else:
            details_view_df = df
        self.__stats.compute(details_view_df)
        return self.__stats.get_statistics()

    def generate_summary(self, event_df: pd.DataFrame) -> ViewData:
        summary_view = super().generate_summary(event_df)
        if summary_view.data.empty:
            return summary_view
        summary_data_df = summary_view.data.rename(index={redc.VALUE: svdc.AGGREGATED})
        summary_view.data = summary_data_df
        return summary_view

    @staticmethod
    def __is_raw_emon_dataframe(df: Union[RawEmonDataFrame, pd.DataFrame]):
        try:
            return all(df.columns == redc.COLUMNS)
        except ValueError:
            return False


class _SocketDataView(_DataView):
    """
    View definition for the Socket summary and detail views
    """

    def __init__(self, config: ViewAttributes):
        super().__init__(config,
                         summary_group_by=[redc.NAME, redc.SOCKET],
                         aggregator_columns=[redc.NAME, redc.SOCKET, redc.UNIT, redc.TSC, redc.VALUE],
                         aggregator_group_by=[redc.NAME, redc.SOCKET, redc.UNIT],
                         details_group_by=[redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME, redc.SOCKET],
                         details_index=[redc.GROUP, redc.TIMESTAMP, redc.SOCKET],
                         device_filter=['SYSTEM'] + config.device.exclusions,
                         device_filter_mode='exclude'
                         )

    def _override_constant_values(self, symbol_table):
        # Adjust the value of the "system.socket_count" constant to 1 to properly compute socket-level metrics
        updated_system_information = symbol_table.copy()
        updated_system_information['system.socket_count'] = 1
        return updated_system_information


class _CoreDataView(_DataView):
    """
    View definition for the Core summary and detail views
    """

    def __init__(self, config: ViewAttributes):
        if config.show_modules:
            super().__init__(config,
                             summary_group_by=[redc.NAME, redc.SOCKET, redc.MODULE, redc.CORE],
                             aggregator_columns=[redc.NAME, redc.SOCKET, redc.MODULE, redc.CORE, redc.UNIT, redc.TSC,
                                                 redc.VALUE],
                             aggregator_group_by=[redc.NAME, redc.SOCKET, redc.MODULE, redc.CORE, redc.UNIT],
                             details_group_by=[redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME, redc.SOCKET,
                                               redc.MODULE, redc.CORE],
                             details_index=[redc.GROUP, redc.TIMESTAMP, redc.SOCKET, redc.MODULE, redc.CORE],
                             device_filter=[config.device.type_name]
                             )
        else:  # Retain backward compatibility
            super().__init__(config,
                             summary_group_by=[redc.NAME, redc.SOCKET, redc.CORE],
                             aggregator_columns=[redc.NAME, redc.SOCKET, redc.CORE, redc.UNIT, redc.TSC, redc.VALUE],
                             aggregator_group_by=[redc.NAME, redc.SOCKET, redc.CORE, redc.UNIT],
                             details_group_by=[redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME, redc.SOCKET, redc.CORE],
                             details_index=[redc.GROUP, redc.TIMESTAMP, redc.SOCKET, redc.CORE],
                             device_filter=[config.device.type_name]
                             )


class _ThreadDataView(_DataView):
    """
    View definition for the Thread summary and detail views
    """

    def __init__(self, config: ViewAttributes):
        if config.show_modules:
            super().__init__(config,
                             summary_group_by=[redc.NAME, redc.UNIT, redc.SOCKET, redc.MODULE, redc.CORE, redc.THREAD],
                             aggregator_columns=[redc.NAME, redc.SOCKET, redc.MODULE, redc.CORE, redc.THREAD, redc.UNIT,
                                                 redc.TSC, redc.VALUE],
                             aggregator_group_by=[redc.NAME, redc.SOCKET, redc.MODULE, redc.CORE, redc.THREAD,
                                                  redc.UNIT],
                             details_group_by=[redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME, redc.UNIT],
                             details_index=[redc.GROUP, redc.TIMESTAMP, redc.UNIT, redc.SOCKET,
                                            redc.MODULE, redc.CORE, redc.THREAD],
                             device_filter=[config.device.type_name]
                             )
        else:  # Retain backward compatibility
            super().__init__(config,
                             summary_group_by=[redc.NAME, redc.UNIT, redc.SOCKET, redc.CORE, redc.THREAD],
                             aggregator_columns=[redc.NAME, redc.SOCKET, redc.CORE, redc.THREAD, redc.UNIT,
                                                 redc.TSC, redc.VALUE],
                             aggregator_group_by=[redc.NAME, redc.SOCKET, redc.CORE, redc.THREAD, redc.UNIT],
                             details_group_by=[redc.GROUP, redc.TIMESTAMP, redc.TSC, redc.NAME, redc.UNIT],
                             details_index=[redc.GROUP, redc.TIMESTAMP, redc.UNIT, redc.SOCKET,
                                            redc.CORE, redc.THREAD],
                             device_filter=[config.device.type_name]
                             )


class _UncoreDataView(_DataView):
    """
    Generic view definition for all uncore unit level views
    """

    def __init__(self, config: ViewAttributes):
        super().__init__(config,
                         summary_group_by=[redc.NAME, redc.UNIT, redc.SOCKET],
                         aggregator_columns=[redc.NAME, redc.SOCKET, redc.UNIT, redc.TSC, redc.VALUE],
                         aggregator_group_by=[redc.NAME, redc.SOCKET, redc.UNIT],
                         details_group_by=[redc.GROUP, redc.TIMESTAMP, redc.SOCKET, redc.TSC, redc.NAME, redc.UNIT],
                         details_index=[redc.GROUP, redc.TIMESTAMP, redc.UNIT, redc.SOCKET],
                         device_filter=[config.device.type_name],
                         )

    def _override_constant_values(self, symbol_table):
        # Similar to the _override_constant_values method in _SocketDataView
        # Adjust the value of "system.socket_count", "system.cha_count", and "chas_per_socket" constants to 1 to
        # properly compute uncore-unit-level metrics
        updated_system_information = symbol_table.copy()
        updated_system_information['system.socket_count'] = 1
        per_socket_symbols = list(filter(lambda x: 'per_socket' in x and 'system.' in x,
                                         list(updated_system_information.keys())))
        for symbol in per_socket_symbols:
            updated_system_information[symbol] = 1
        return updated_system_information


class _NullNormalizer(Normalizer):
    """
    A "do nothing" normalizer, used as default when normalization is not required
    """

    def __init__(self):
        pass

    def normalize(self, df: pd.DataFrame, event_axis: str = 'columns') -> pd.DataFrame:
        return df


class _NullMetricComputer(MetricComputer):
    """
    a "do nothing" metric computer, used as default when metric computation is not required
    """

    def __init__(self, metric_definition_list: List[MetricDefinition], symbol_table: Dict = None):
        pass

    def compute_metric(self, df: pd.DataFrame,
                       constant_values: Dict[str, str] = None,
                       calculate_block_level: bool = False,
                       group_index_name: str = redc.GROUP,
                       timestamp_index_name: str = redc.TIMESTAMP) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def symbol_table(self):
        return {}
