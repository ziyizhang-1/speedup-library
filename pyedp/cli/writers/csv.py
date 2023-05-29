#  Copyright 2021 Intel Corporation
#  This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
#  express license under which they were provided to you (License). Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
#  prior written permission.
#
#  This software and the related documents are provided as is, with no express or implied warranties, other than those
#  that are expressly stated in the License.
#
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from cli.writers.base import OutputWriter
from pedp import ViewAttributes


class CSVWriter(OutputWriter):
    """
    Write Pandas DataFrame content to CSV
    """

    def __init__(self, csv_directory_path: Path):
        """
        Initialize an EDP Output CSV Writer

        :param csv_directory_path: location in which to create CSV files
        """
        self.__csv_files = {}
        self.__csv_directory = csv_directory_path

    def write(self, title: str, content: pd.DataFrame, attributes: Optional[ViewAttributes] = None, mode: str = None, print_header: bool = None) -> None:
        """
        Write/append the content of a Pandas DataFrame to a CSV file.
        The file is created if it does not exist.

        :param title: CSV file name (basename, without path)
        :param content: DataFrame to write
        :param attributes: view attributes (optional)
        """
        csv_info = self.__get_csv_file(title)
        if mode is None:
            mode = 'w' if csv_info.start_row == 0 else 'a'
        if print_header is None:
            print_header = True if csv_info.start_row == 0 else False
        # Pandas "to_csv" function works significantly faster with larger chunk size (default is None).
        # chunksize=200 seems to yield a good speed-up with a reasonable memory consumption.
        content.to_csv(csv_info.path, header=print_header, index=False, mode=mode, date_format='%m/%d/%Y %H:%M:%S.%f',
                       chunksize=200)
        csv_info.start_row += len(content)

    def close(self):
        """
        Finalize and close the CSV file. Subsequent operations on this CSV Writer object are not permitted.
        """
        # Function defined to override the generic documentation of the base class.
        # Pandas will automatically close and finalize the generated CSV files during the `write` operation.
        pass

    def __get_csv_file(self, name: str) -> '_CSVInfo':
        """
        Allocate a new CSV file name for the view, or return an existing file if already allocated

        :param name: must be in the format "{EDP view name}_{EDP view type}",
                     e.g. "socket_summary", "system_details", etc.
        """
        if name not in self.__csv_files:
            csv_path = self.__csv_directory / Path(f'{name}.csv')
            self.__csv_files[name] = self._CSVInfo(name, csv_path)
        return self.__csv_files[name]

    @dataclass
    class _CSVInfo:
        name: str
        path: Path
        start_row: int = 0
