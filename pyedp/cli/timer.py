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
import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.__start = time.perf_counter()
        self.__elapsed = 0

    @property
    def start_time(self):
        return self.__start

    @property
    def elapsed_time(self):
        return self.__elapsed if self.__elapsed else time.perf_counter() - self.__start

    def stop(self):
        self.__elapsed = time.perf_counter() - self.__start

    def __str__(self):
        return f'{self.elapsed_time:.2f} seconds'


@contextmanager
def timer():
    t = Timer()
    try:
        yield t
    finally:
        t.stop()
