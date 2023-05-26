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
from typing import Optional

import pandas as pd

from pedp import ViewAttributes


class OutputWriter(ABC):
    """
    EDP Output writer abstract base class (ABC)
    """

    @abstractmethod
    def write(self, title: str, content: pd.DataFrame, attributes: Optional[ViewAttributes] = None) -> None:
        """
        Write DataFrame content. Output type is determined by subclasses.

        :param title: the title/name of the content
        :param content: the content to write, as a Pandas DataFrame
        :param attributes: view attributes (optional)
        """
        pass

    @abstractmethod
    def close(self):
        """
        Finalize and close the EDP Output Writer
        """
        pass
