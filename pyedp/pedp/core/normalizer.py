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
import pandas as pd
from pedp.core.types import RawEmonDataFrameColumns as redc


class Normalizer:
    def __init__(self, ref_tsc):
        self.ref_tsc = ref_tsc
        self.__events_to_exclude = ['$samplingTime', '$processed_samples']

    def normalize(self, df: pd.DataFrame, event_axis: str = 'columns') -> pd.DataFrame:
        """
        Computes normalized event count

        @param df: data frame containing data to normalize
        @param event_axis: axis where event names exist, must be either 'columns' or 'index'

        @return a copy of df where the "value" column is updated to contain normalized values
        """
        self.__validate_event_axis(event_axis)
        rows_to_not_normalize, rows_to_normalize = self.__split_df(df.copy(), event_axis)
        rows_to_normalize[redc.VALUE] = rows_to_normalize[redc.VALUE] * self.ref_tsc / rows_to_normalize[redc.TSC]
        normalized_df = pd.concat([rows_to_normalize, rows_to_not_normalize])
        return normalized_df

    def __split_df(self, df: pd.DataFrame, event_axis: str):
        rows_to_not_normalize = []
        if redc.NAME not in df.columns and not all(metric in df.index for metric in self.__events_to_exclude):
            return None, df  # all rows can be normalized
        if event_axis == 'columns' and redc.NAME in df.columns:
            rows_to_not_normalize = df[redc.NAME].isin(self.__events_to_exclude)
        elif event_axis == 'index' and all(metric in df.index for metric in self.__events_to_exclude):
            rows_to_not_normalize = self.__events_to_exclude
        rows_to_not_normalize = df.loc[rows_to_not_normalize]
        rows_to_normalize = df.loc[list(set(df.index).difference(set(rows_to_not_normalize.index)))]
        return rows_to_not_normalize, rows_to_normalize

    @staticmethod
    def __validate_event_axis(event_axis: str):
        if event_axis not in ['columns', 'index']:
            raise ValueError("'event_axis' argument must be either 'columns' or 'index'")
