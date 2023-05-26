#  Copyright 2021 Intel Corporation
#  This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
#  express license under which they were provided to you (License). Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
#  prior written permission.
#
#  This software and the related documents are provided as is, with no express or implied warranties, other than those
#  that are expressly stated in the License.
#
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Dict, Tuple
from contextlib import AbstractContextManager

import pandas as pd

from cli import __version__
from cli.writers.base import OutputWriter
from pedp.core.views import ViewAggregationLevel, ViewType, ViewData, ViewAttributes, ViewCollection


class ViewWriter(AbstractContextManager):
    """
    Format summary and details EDP View DataFrames and write them to multiple writers.

    `ViewWriter` is a context manager and should be used with a `with` statement to guarantee that all writers
    are properly closed.

    """

    show_modules: bool

    def __init__(self,
                 show_modules: bool,
                 writers: Union[OutputWriter, List[OutputWriter]]):
        """
        Initialize the View Writer

        :param writers: a single writer object, or a list of writer objects, to use for outputting data.
        """

        ViewWriter.show_modules = show_modules
        self.__writers = writers if type(writers) is list else [writers]
        self.__formatters: Dict[Tuple[ViewAggregationLevel, ViewType], ViewWriter._ViewFormatter] = {}
        self.__initialize_view_formatters()

    def __exit__(self, exc_type, exc_value, traceback):
        for writer in self.__writers:
            writer.close()

    def write(self,
              views: List[ViewData],
              first_sample: int,
              last_sample: int) -> None:
        """
        Write EDP views to output.

        :param views: the EDP views to write
        :param first_sample: the number of the first sample processed
        :param last_sample: the number of the last sample processed
        """
        for view in views:
            if view.data is None or view.data.empty:
                continue
            formatter = self.__formatters[(view.attributes.aggregation_level, view.attributes.view_type)]
            formatted_content = formatter.format(view.data, first_sample, last_sample)
            self.__write(view.attributes, formatted_content)

    def __write(self, view_attributes: ViewAttributes, view_content: pd.DataFrame) -> None:
        for writer in self.__writers:
            writer.write(f'{view_attributes.view_name}', view_content, view_attributes)

    def __initialize_view_formatters(self):
        def create_view_formatter(agg_level: ViewAggregationLevel, view_type: ViewType):
            formatters = {
                (ViewAggregationLevel.SYSTEM, ViewType.SUMMARY): self._SystemSummaryViewFormatter,
                (ViewAggregationLevel.SYSTEM, ViewType.DETAILS): self._SystemDetailsViewFormatter,
                (ViewAggregationLevel.SOCKET, ViewType.SUMMARY): self._SocketSummaryViewFormatter,
                (ViewAggregationLevel.SOCKET, ViewType.DETAILS): self._SocketDetailsViewFormatter,
                (ViewAggregationLevel.CORE, ViewType.SUMMARY): self._CoreSummaryViewFormatter,
                (ViewAggregationLevel.CORE, ViewType.DETAILS): self._CoreDetailsViewFormatter,
                (ViewAggregationLevel.THREAD, ViewType.SUMMARY): self._ThreadSummaryViewFormatter,
                (ViewAggregationLevel.THREAD, ViewType.DETAILS): self._ThreadDetailsViewFormatter,
                (ViewAggregationLevel.UNCORE, ViewType.SUMMARY): self._UncoreSummaryViewFormatter,
                (ViewAggregationLevel.UNCORE, ViewType.DETAILS): self._UncoreDetailsViewFormatter,
            }
            return formatters[(agg_level, view_type)]()

        self.__formatters = {(agg_level, view_type): create_view_formatter(agg_level, view_type)
                             for agg_level in ViewAggregationLevel
                             for view_type in [ViewType.SUMMARY, ViewType.DETAILS]}

    class _ViewFormatter(ABC):
        """
        Abstract base class for view formatters
        """

        def __init__(self, rename_columns_func: Callable):
            self._rename_columns_func = rename_columns_func

        @abstractmethod
        def format(self, view_content: pd.DataFrame, first_sample: int, last_sample: int) -> pd.DataFrame:
            """
            Format an EDP view for output

            :param view_content: view DataFrame to format
            :param first_sample: the number of the first sample processed
            :param last_sample: the number of the last sample processed

            :return: a new formatted DataFrame
            """
            pass

    class _SummaryViewFormatter(_ViewFormatter):
        """
        Format Summary View DataFrames to the desired EDP output format (e.g., change column headers,
        add columns, etc.)
        """

        def format(self, view_content: pd.DataFrame, first_sample: int, last_sample: int) -> pd.DataFrame:
            """
            Format an EDP summary view for output

            :param view_content: summary view DataFrame to format
            :param first_sample: the number of the first sample processed
            :param last_sample: the number of the last sample processed

            :return: a new formatted DataFrame
            """
            formatted_content = view_content.transpose()
            formatted_content.columns = self._rename_columns_func(formatted_content.columns)
            first_column_title = f'(EDP {__version__}) name (sample #{first_sample} - #{last_sample})'
            formatted_content.reset_index(inplace=True)
            formatted_content.rename(columns={'index': first_column_title}, inplace=True)
            return formatted_content

    class _DetailsViewFormatter(_ViewFormatter):
        """
        Format Details View DataFrames to the desired EDP output format (e.g., change column headers, add columns, etc.)
        """

        def format(self, view_content: pd.DataFrame, first_sample: int, last_sample: int) -> pd.DataFrame:
            """
            Format an EDP details view for output

            :param view_content: details view DataFrame to format
            :param first_sample: the number of the first sample processed
            :param last_sample: the number of the last sample processed

            :return: a new formatted DataFrame
            """
            formatted_content = view_content
            if view_content.index.names[1:]:
                formatted_content = view_content.unstack(view_content.index.names[1:])
            # Add "sample#" to the index
            formatted_content = formatted_content.reset_index().set_index('timestamp', append=True)
            formatted_content.index.set_names('#sample', level=0, inplace=True)
            formatted_content.columns = self._rename_columns_func(formatted_content.columns)
            # Update the value of the "#sample" column to reflect the actual sample number being written
            formatted_content = formatted_content.reset_index()
            formatted_content['#sample'] += (last_sample - len(formatted_content) + 1)
            return formatted_content

    class _SystemSummaryViewFormatter(_SummaryViewFormatter):
        """
        Define the parameters and logic for formatting the System Summary view for output
        """

        def __init__(self):
            super().__init__(rename_columns_func=lambda x: x)

    class _SystemDetailsViewFormatter(_DetailsViewFormatter):
        """
        Define the parameters and logic for formatting the System Details view for output
        """

        def __init__(self):
            super().__init__(rename_columns_func=lambda x: x)

    class _SocketSummaryViewFormatter(_SummaryViewFormatter):
        """
        Define the parameters and logic for formatting the Socket Summary view for output
        """

        def __init__(self):
            super().__init__(rename_columns_func=lambda cols: [f'socket {socket:.0f}' for socket in cols])

    class _SocketDetailsViewFormatter(_DetailsViewFormatter):
        """
        Define the parameters and logic for formatting the Socket Details view for output
        """

        def __init__(self):
            super().__init__(rename_columns_func=lambda cols: [f'{name} (socket {socket:.0f})'
                                                               for name, socket in cols])

    class _CoreSummaryViewFormatter(_SummaryViewFormatter):
        """
        Define the parameters and logic for formatting the Core Summary view for output
        """

        def __init__(self):
            if ViewWriter.show_modules:
                super().__init__(
                    rename_columns_func=lambda cols: [f'socket {socket:.0f} module {module:.0f} core {core:.0f}'
                                                      for socket, module, core in cols])
            else:  # retain backward compatibility
                super().__init__(rename_columns_func=lambda cols: [f'socket {socket:.0f} core {core:.0f}'
                                                                   for socket, core in cols])

    class _CoreDetailsViewFormatter(_DetailsViewFormatter):
        """
        Define the parameters and logic for formatting the Core Details view for output
        """

        def __init__(self):
            if ViewWriter.show_modules:
                super().__init__(rename_columns_func=lambda cols: [f'{name} (socket {socket:.0f} module {module:.0f} '
                                                                   f'core {core:.0f})'
                                                                   for name, socket, module, core in cols])
            else:  # retain backward compatibility
                super().__init__(rename_columns_func=lambda cols: [f'{name} (socket {socket:.0f} core {core:.0f})'
                                                                   for name, socket, core in cols])

    class _ThreadSummaryViewFormatter(_SummaryViewFormatter):
        """
        Define the parameters and logic for formatting the Thread Summary view for output
        """

        def __init__(self):
            if ViewWriter.show_modules:
                super().__init__(rename_columns_func=lambda cols: [f'cpu {unit} (S{socket:.0f}M{module:.0f}C{core:.0f}'
                                                                   f'T{thread:.0f})'
                                                                   for unit, socket, module, core, thread in cols])
            else:  # retain backward compatibility
                super().__init__(rename_columns_func=lambda cols: [f'cpu {unit} (S{socket:.0f}C{core:.0f}T{thread:.0f})'
                                                                   for unit, socket, core, thread in cols])

    class _ThreadDetailsViewFormatter(_DetailsViewFormatter):
        """
        Define the parameters and logic for formatting the Thread Details view for output
        """

        def __init__(self):
            super().__init__(rename_columns_func=lambda cols: [f'{c[0]} (cpu {c[1]})' for c in cols])


    class _UncoreSummaryViewFormatter(_SummaryViewFormatter):
        """
        Define the parameters and logic for formatting the Uncore Summary view for output
        """

        def __init__(self):
            super().__init__(rename_columns_func=lambda cols: [f'unit {unit} (S{socket:.0f})'
                                                               for unit, socket in cols])

        def format(self, view_content, first_sample, last_sample):
            view_content = view_content.sort_index(level=1)
            formatted_content = super().format(view_content, first_sample, last_sample)
            return formatted_content

    class _UncoreDetailsViewFormatter(_DetailsViewFormatter):
        """
        Define the parameters and logic for formatting the Uncore Details view for output
        """

        def __init__(self):
            super().__init__(rename_columns_func=lambda cols: [f'{c[0]} (unit {c[1]} socket {c[2]:.0f})' for c in cols])

        def format(self, view_content: pd.DataFrame, first_sample: int, last_sample: int) -> pd.DataFrame:
            view_content = view_content.sort_index(level=2)
            formatted_content = super().format(view_content, first_sample, last_sample)
            return formatted_content

class DataMerger:
    """
    Used to write detail views from each Parition into a final details view.
    Utilizes view_writer and view_collection to write each view.
    """

    def __init__(self, view_collection: ViewCollection, view_writer: ViewWriter):
        self.__view_collection = view_collection
        self.__view_writer = view_writer
        self.__detail_views = {attributes.view_name: ViewData(attributes, pd.DataFrame()) for attributes in
                               self.__view_collection.views}

    @property
    def views(self):
        return list(self.__detail_views.values())

    def write_to_detail_views(self, detail_views: List[ViewData], partition: 'Partition'):
        """

        @param detail_views:
        @param partition: Partition object used to determine the first and last sample of the details_view partition
        @return: writes all detail views to storage depending on their first and last sample
        """
        self.__view_writer.write(detail_views, first_sample=partition.first_sample,
                                 last_sample=partition.last_sample)
