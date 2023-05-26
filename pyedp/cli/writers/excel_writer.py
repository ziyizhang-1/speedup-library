import argparse

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xlsxwriter

from cli.timer import timer
from pedp import ViewAttributes, ViewType, ViewCollection


class ExcelWriter:
    """
    A xlsxwriter wrapper to create EDP Excel file from pre-generated text and CSV files
    """

    def __init__(self, cmd_line_args: argparse.Namespace):
        self.__args = cmd_line_args
        self.__include_details = False if self.__args.no_detail_views else True
        self.__include_charts = True if self.__include_details and self.__args.chart_format_file_path else False
        self.__csv_path = Path(self.__args.output_file_path).parent
        self.__workbook = xlsxwriter.Workbook(self.__args.output_file_path,
                                              {
                                                  'strings_to_numbers': True,
                                                  'constant_memory': True,
                                                  'use_zip64': True,
                                              })
        self.__sheet_order = {}
        self.__sheet_view_name_map = {}

    def get_sheet_and_format(self, name, tab_color, font_size, num_format, font_name='Calibri'):
        """
        Create a new worksheet and a format object

        :param name: worksheet name
        :param tab_color: worksheet tab color
        :param font_size: worksheet font size
        :param num_format: format string to use for numbers
        :param font_name: worksheet font size

        :return: a (worksheet, format) tuple
        """
        worksheet = self.__workbook.add_worksheet(name)
        worksheet.set_tab_color(tab_color)
        fmt = self.__workbook.add_format({'num_format': num_format, 'font_size': font_size, 'font_name': font_name})
        return worksheet, fmt

    def import_file(self, file, name):
        """
        Import the content of the specified file to a new Excel worksheet

        :param file: path of file to import
        :param name: worksheet name
        """
        if not file or not Path(file).is_file():
            print(f' WARNING: {name} file does not exist.')
            return

        print(f'     importing {name}')
        with open(file) as f:
            sheet, fmt = self.get_sheet_and_format(name, '#001400', 10, '#,##0')
            for row_index, line in enumerate(f, 0):
                for col_index, col in enumerate(line.split('\t'), 0):
                    sheet.write(row_index, col_index, col, fmt)

        print('     imported {name}')

    def import_summary(self, file, name, tab_color, font_size, num_format, font_name):
        """
        Import the content of an EDP summary view (in CSV) to Excel

        :param file: path of file to import
        :param name: worksheet name
        :param tab_color: worksheet tab color
        :param font_size: worksheet font size
        :param num_format: format string to use for numbers
        :param font_name: worksheet font size
        """
        if not file or not Path(file).is_file():
            return

        sheet, fmt = self.get_sheet_and_format(name, tab_color, font_size, num_format, font_name)
        row_length = 0
        with open(file) as f:
            for row_index, line in enumerate(f, 0):
                for col_index, value in enumerate(line.split(','), 0):
                    sheet.write(row_index, col_index, value, fmt)
                row_length += 1
        sheet.set_column(0, 0, 40)
        sheet.set_column(1, row_length, 22.36)

    def import_view(self, view_attributes: ViewAttributes):
        """
        Import EDP view to Excel
        :param view_attributes: attributes for the view to import
        """
        view_type = view_attributes.view_type
        if view_attributes.view_type == ViewType.SUMMARY and 'per_txn' in view_attributes.view_name:
            self.__import_tps_summary_view(view_attributes)
        elif view_type == ViewType.SUMMARY:
            self.__import_summary_view(view_attributes)
        elif view_type == ViewType.DETAILS and self.__include_details:
            self.__import_details_for_view(view_attributes)
        else:
            view_name = view_attributes.view_name
            raise ValueError(f'Unsupported view type, {view_type}, for {view_name}')

    def __prepare_import(self, view_attributes: ViewAttributes, sheet_postfix=''):
        csv = self.__csv_path / Path(f'{view_attributes.view_name}.csv')
        sheet_name = self.__get_excel_sheet_name(view_attributes) + sheet_postfix
        self.__sheet_view_name_map[view_attributes.view_name] = self.__sheet_view_name_map.get(
            view_attributes.view_name, [])
        self.__sheet_view_name_map[view_attributes.view_name].append(sheet_name)
        print(f'     importing {sheet_name}...')
        return csv, sheet_name

    def __import_details_for_view(self, view_attributes: ViewAttributes):
        headers, num_samples = self.__import_details_view(view_attributes)
        if self.__include_charts:
            self.__import_details_chart(view_attributes, headers, num_samples)

    def __import_summary_view(self, view_attributes: ViewAttributes):
        csv, sheet_name = self.__prepare_import(view_attributes)
        self.import_summary(csv, sheet_name, '#006300', 10, '#,##0.0000', 'Courier New')

    def __import_tps_summary_view(self, view_attributes: ViewAttributes):
        csv, sheet_name = self.__prepare_import(view_attributes, ' (per-txn)')
        self.import_summary(csv, sheet_name, '#000000', 10, '#,##0.0000', 'Courier New')

    def __import_details_view(self, view_attributes: ViewAttributes):
        csv, sheet_name = self.__prepare_import(view_attributes)
        num_samples = 0
        headers = []
        row_length = 0
        if (not self.__args.no_detail_views) and csv and Path(csv).is_file():
            sheet, fmt = self.get_sheet_and_format(sheet_name, '#630000', 8, '#,##0.0000')
            first_col_format = self.__workbook.add_format(
                {'num_format': '#,##0', 'font_size': 8, 'font_name': 'Calibri'})
            with open(csv) as f:
                for row_index, line in enumerate(f, 0):
                    row = line.split(',')
                    sheet.write(row_index, 0, row[0], first_col_format)
                    if row_index == 0:
                        headers = row

                    if row[1] != 'timestamp':  # and not self.__args.timestamp_in_chart: - TODO: enable?
                        # Remove trailing zeros in timestamp
                        row[1] = row[1].rstrip('000')
                    for col_index in range(1, len(row)):
                        # Skip empty and NaN values to reduce the file size. NaN values also cause troubles with charts
                        if row[col_index] == '' or row[col_index] == 'NaN':
                            continue
                        sheet.write(row_index, col_index, row[col_index], fmt)
                    row_length = len(row)
                    num_samples += 1
            sheet.set_column(20, row_length, 12.45)
            return headers, num_samples

    def __import_details_chart(self, view_attributes: ViewAttributes, headers: List[str], num_samples: int):
        """
        Create charts for the specified view

        :param view_attributes: the view type for which to create charts: 'system', 'socket', 'core' or 'thread'
        :param headers: list containing the Details view headers ('#sample', 'timestamp', event and metric names)
        :param num_samples: number of samples in the Details view
        """
        device = view_attributes.device
        if not self.__args.chart_format_file_path or \
                device.type_name in device.exclusions or \
                device.type_name not in self.__args.chart_format_file_path:
            return

        sheet_name = self.__get_excel_sheet_name(view_attributes)
        chart_name = self.__get_excel_chart_name(view_attributes)
        self.__sheet_view_name_map[view_attributes.view_name] = self.__sheet_view_name_map.get(
            view_attributes.view_name, [])
        self.__sheet_view_name_map[view_attributes.view_name].append(chart_name)
        print(f'        plotting charts for {chart_name}: ', end='')
        chart_sheet, chart_format = self.get_sheet_and_format(chart_name, '#000063', 10, None, 'Calibri')

        # TODO: enable "timestamp_in_chart" option
        # cat_col = 'timestamp' if self.__args.timestamp_in_chart else '#sample'
        cat_col = '#sample'
        categories = []
        for col_index, header in enumerate(headers, 0):
            if header == cat_col:
                categories = [sheet_name, 1, col_index, num_samples, col_index]
                break

        row = 1
        x_offset = 10
        col = 0

        num_charts = 0
        with timer() as number_of_seconds:
            with open(self.__args.chart_format_file_path[device.type_name]) as f:
                for metric in f:
                    metric = metric.strip()
                    if metric == '':
                        row += 18
                        col = 0
                        x_offset = 10
                        continue

                    chart_not_created = True
                    for col_index, header in enumerate(headers, 0):
                        if metric == header or (metric + ' ') in header:
                            values = [sheet_name, 1, col_index, num_samples, col_index]

                            if chart_not_created:
                                chart = self.__workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
                                chart_not_created = False

                            chart.add_series({
                                'name': f'{header}',
                                'categories': categories,
                                'values': values,
                            })
                    # Chart w/o series results in error when writing to excel (this is by design of the library)
                    if chart_not_created:
                        continue

                    chart.set_legend({'position': 'bottom'})
                    chart.set_title({'name': metric, 'name_font': {'size': 11}})
                    chart.set_size({'width': 408, 'height': 343.68, 'x_offset': x_offset})
                    chart_sheet.insert_chart(row, col, chart)
                    col += 6
                    x_offset += 35
                    num_charts += 1
                    print('+', end='')
            print(f'\n        {num_charts} charts plotted in {number_of_seconds}')

    def sort_sheets(self, view_collection: ViewCollection):
        """
        Sort the workbook sheets to match the expected EDP Excel file format
        """
        views = view_collection.views
        sheet_order = {}
        sheets = []
        for idx, view in enumerate(views):
            view_sheet_names = self.__sheet_view_name_map[view.view_name]
            for sheet_name in view_sheet_names:
                sheets.append(sheet_name)
                sheet_order[sheet_name] = [idx]
                sheet_order.update(self.__get_sheet_order(sheet_order, view, sheet_name))
        sheet_order = pd.Series(sheet_order).sort_values()
        order = []
        for sheet in sheets:
            order.append(sheet_order[sheet])
        self.__workbook.worksheets_objs = [self.__workbook.sheetnames[name] for name in sheet_order.index]

    def close(self):
        """
        Save and close the output Excel file. After calling this method, additional operations on the object are
        not permitted.
        """
        self.__workbook.close()

    @staticmethod
    def __get_device_prefix(device: str):
        return '' if device == 'core' else device + ' '

    def __get_excel_sheet_name(self, view_attributes: ViewAttributes):
        device_prefix = ExcelWriter.__get_device_prefix(view_attributes.device.label)
        aggregation_level = view_attributes.aggregation_level.name.lower()
        view_type = 'details ' if view_attributes.view_type.name.lower() == 'details' else ''
        sheet_name = f'{device_prefix}{view_type}{aggregation_level} view'
        sheet_name = sheet_name.strip()
        return sheet_name

    def __get_sheet_order(self, sheet_order: Dict[str, List[int]],
                          view_attributes: ViewAttributes,
                          sheet_name: str) -> Dict:
        """
        Given a view, will provide the proper order in which the view should appear in the final Excel output,
        based on the following assumptions about order:
        1. Summary views (including per-txn)
        2. chart views
        3. detail views

        The ordering becomes a length 2 tuple, where the first index is the order in which the view is produced,
        and the second index is based on the assumptions listed above. This ensures, for example, that system sheets
        will come before socket sheets because system views are generated before socket views, and summary sheets
        come before detail sheets.

        @param sheet_order: the current ordering of Excel sheets
        @param view_attributes: attributes for the view being added as a sheet in Excel
        @param sheet_name: the name of the sheet being added in Excel
        @return: an updated sheet ordering with a new sheet added
        """
        if view_attributes.view_type == ViewType.SUMMARY:
            sheet_order[sheet_name].insert(0, 1)
        elif 'chart' in sheet_name:
            sheet_order[sheet_name].insert(0, 2)
        elif view_attributes.view_type == ViewType.DETAILS:
            sheet_order[sheet_name].insert(0, 3)
        sheet_order[sheet_name] = tuple(sheet_order[sheet_name])
        return sheet_order

    @staticmethod
    def __get_excel_chart_name(view_attributes: ViewAttributes):
        device_prefix = ExcelWriter.__get_device_prefix(view_attributes.device.label)
        aggregation_level = view_attributes.aggregation_level.name.lower()
        return f'{device_prefix}chart {aggregation_level} view'.strip()


def write_csv_data_to_excel(cmd_line_args, view_collection: ViewCollection):
    with timer() as number_of_seconds:
        excel_writer = ExcelWriter(cmd_line_args)

        for view in view_collection.views:
            excel_writer.import_view(view)

        excel_writer.sort_sheets(view_collection)
        excel_writer.close()

        print(f'Wrote Excel file in {number_of_seconds}')
        print(f'Output written to: {cmd_line_args.output_file_path}')
