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
from typing import NewType

import pandas as pd

from pedp.core.types import SummaryViewDataFrameColumns

StatisticsDataFrame = NewType('StatisticsDataFrame', pd.DataFrame)


class StatisticsDataFrameColumns:
    MIN = SummaryViewDataFrameColumns.MIN
    MAX = SummaryViewDataFrameColumns.MAX
    COUNT = 'count'
    SUM = 'sum'
    PERCENTILE = SummaryViewDataFrameColumns.PERCENTILE
    VARIATION = SummaryViewDataFrameColumns.VARIATION

    COLUMNS = [MIN, MAX, PERCENTILE, COUNT, SUM, VARIATION]