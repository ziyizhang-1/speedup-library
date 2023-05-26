#  Copyright 2022 Intel Corporation
#  This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
#  express license under which they were provided to you (License). Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
#  prior written permission.
#
#  This software and the related documents are provided as is, with no express or implied warranties, other than those
#  that are expressly stated in the License.
#
#
from typing import Dict

from pedp import (
    ViewData,
    ViewAggregationLevel,
    MetricComputer,
    MetricDefinition,
    SummaryViewDataFrameColumns as svdf
)


class TPSViewGenerator:
    """
    Generate Transactions Per Second (TPS) views for EDP Summary views
    """
    def __init__(self, tps_value: int):
        """
        Initialize the TPS view generator

        :param tps_value: transactions per second value to use for TPS computations
        """
        self.__tps = tps_value

    def generate_summaries(self, summary_views: Dict[str, ViewData]) -> Dict[str, ViewData]:
        """
        Generate a TPS view for each of the input views

        :param summary_views: EDP Summary views for which to generate TPS values

        :return: a list of EDP Summary views corresponding to `summary_views`, where metric and event values are
                 replaced with TPS values
        """
        tps_summary_views = {}
        for _, view in summary_views.items():
            tps_summary_views.update(self.__compute_tps_events_and_metrics(view))
        return tps_summary_views

    def __compute_tps_events_and_metrics(self, view: ViewData) -> Dict[str, ViewData]:
        tps_metric_computer = _TPSMetricComputer(view, self.__tps)
        return tps_metric_computer.generate_tps_view()


class _TPSMetricComputer:
    """
    Computes Transactions Per Second (TPS) values for events and metrics
    """
    def __init__(self, view: ViewData, tps: int):
        """
        Initialize the TPS metric computer

        :param view: EDP Summary view for which to compute TPS values
        :param tps: transactions per second value to use for TPS computations
        """
        self.__view = view
        self.__tps = tps
        self.metric_defs = [m.definition for m in self.__view.attributes.metric_computer.compiled_metrics]
        self.__initialize_metric_to_tps_map()
        self.__initialize_tps_metric_computer()
        self.__initialize_tps_events_computer()

    def generate_tps_view(self) -> Dict[str, ViewData]:
        """
        Generate the EDP TPS View

        :return: an EDP TPS Summary view
        """
        tps_df = self.__generate_tps_data()
        # TODO: make a unified view name formatter for all views
        device = self.__view.attributes.device
        view_name = f"__edp{device.decorate_label(prefix='_')}_" \
                    f"{self.__view.attributes.aggregation_level.name.lower()}_view_" \
                    f"{self.__view.attributes.view_type.name.lower()}_per_txn"
        new_view_attr = self.__view.attributes.clone(update={'view_name': view_name,
                                                             'metric_computer': self.tps_metrics_computer})
        return {new_view_attr.view_name: ViewData(new_view_attr, tps_df)}

    def __generate_tps_data(self):
        tps_metric_values = self.tps_metrics_computer.compute_metric(self.__view.data)
        tps_event_values = self.tps_events_computer.compute_metric(self.__view.data)
        tps_df = self.__create_tps_data_frame(tps_event_values, tps_metric_values)
        return tps_df

    def __create_tps_data_frame(self, tps_event_values, tps_metric_values):
        tps_df = self.__view.data.copy()
        tps_df.update(tps_metric_values)
        tps_df.update(tps_event_values)
        tps_df = tps_df.rename(columns=self.metric_to_tps_map)
        if self.__view.attributes.aggregation_level == ViewAggregationLevel.SYSTEM:
            # Keep only the 'aggregated' row
            tps_df = tps_df.drop(labels=[label for label in tps_df.index if label != svdf.AGGREGATED], axis='index')
        return tps_df

    def __initialize_metric_to_tps_map(self):
        metrics_supporting_tps = [m for m in self.metric_defs if m.throughput_metric_name != '']
        self.metric_to_tps_map = {m.name: m.throughput_metric_name for m in metrics_supporting_tps}

    def __initialize_tps_events_computer(self):
        # TPS value for events is computed using the following formula: "EVENT_VALUE / tps"
        tps_event_formula_defs = [MetricDefinition(c, '', '', '', f'a/{self.__tps}', {'a': c}, {}, {})
                                  for c in self.__view.data if c not in [m.name for m in self.metric_defs]]
        self.tps_events_computer = MetricComputer(tps_event_formula_defs, {})

    def __initialize_tps_metric_computer(self):
        # TPS value for metrics is computed using the following formula: "METRIC_VALUE * INST_RETIRED.ANY / tps"
        tps_metric_formula_defs = [MetricDefinition(m, '', '', '', f'a*b/{self.__tps}',
                                                    {'a': m, 'b': 'INST_RETIRED.ANY'}, {}, {})
                                   for m in self.metric_to_tps_map.keys()]
        self.tps_metrics_computer = MetricComputer(tps_metric_formula_defs, {})
