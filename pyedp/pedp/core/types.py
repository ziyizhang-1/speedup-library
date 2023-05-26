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
from dataclasses import dataclass, field
from typing import NewType, Dict, List

import pandas as pd


class DeviceType:
    CORE = 'core'
    UNCORE = 'UNCORE'
    SYSTEM = 'SYSTEM'


class Device:
    """
    Device handling class all supported devices ('core', 'bigcore', 'smallcore', 'cha', etc.)
    note: label is blank for non-hybrid cores (ex: 'core') because no core type specifier is
          added to report and filenames for non-hybrid cores.
    @param type_name: gets the type of device, used for filtering an emon df
    """
    valid_device_names: List[str] = []

    def __init__(self,
                 type_name: str = DeviceType.CORE,
                 aggregation_levels: List['ViewAggregationLevel'] = None,
                 metric_computer: 'MetricComputer' = None):
        self.__type_name: str = type_name
        self.__aggregation_levels = aggregation_levels
        self.__label: str = '' if type_name == DeviceType.CORE else self.__type_name
        self.__update_uncore_labels()
        self.__exclusions = [type for type in self.valid_device_names if type != self.__type_name]
        self.__metric_computer = metric_computer

        def validate_preconditions():
            if type_name not in Device.valid_device_names:
                raise ValueError(f'{self.__type_name} is not a valid device.')

        validate_preconditions()

    def __update_uncore_labels(self):
        if 'UNC_' in self.__label:
            self.__label = self.__label.replace('UNC_', '').lower()

    @property
    def aggregation_levels(self):
        return self.__aggregation_levels

    @property
    def metric_computer(self):
        return self.__metric_computer

    @property
    def type_name(self):
        """
        type name for this core
        """
        return self.__type_name

    @property
    def label(self):
        """
        label for this core. Used in filenames, sheet names, etc
        note: label is blank for non-hybrid cores (ex: 'core')
        """
        return self.__label

    @property
    def exclusions(self):
        """
        a list of cores types to filter out when generating reports for this core_type
        """
        return self.__exclusions

    def decorate_label(self, prefix: str = '', postfix: str = ''):
        """
        method for prefixing/postfixing the label with specified characters
        when preparing the label for use in a filename, chart name, stdout statement, etc
        :param prefix: a string to add to the beginning of the label (ex: ' ', '_'
        :param postfix: a string to add to the end of the label (ex: ' ', '_')
        :return: a copy of the decorated label
        """
        return f'{prefix}{self.__label}{postfix}' if self.__label else ''

    @staticmethod
    def set_valid_device_names(unique_devices: List[str]):
        """
        Set valid device names, a static list used for all Device objects
        :param unique_devices: unique/valid devices
                                  exmpled: unique devices parsed from the emon data file
        """
        Device.valid_device_names = unique_devices


RawEmonDataFrame = NewType('RawEmonDataFrame', pd.DataFrame)


class RawEmonDataFrameColumns:
    TIMESTAMP = 'timestamp'
    SOCKET = 'socket'
    DEVICE = 'device'
    CORE = 'core'
    THREAD = 'thread'
    UNIT = 'unit'
    MODULE = 'module'
    TSC = 'tsc'
    GROUP = 'group'
    NAME = 'name'
    VALUE = 'value'

    COLUMNS = [TIMESTAMP, SOCKET, DEVICE, CORE, THREAD, UNIT, MODULE, TSC, GROUP, NAME, VALUE]


SummaryViewDataFrame = NewType('SummaryViewDataFrame', pd.DataFrame)


class SummaryViewDataFrameColumns:
    AGGREGATED = 'aggregated'
    MIN = 'min'
    MAX = 'max'
    PERCENTILE = '95th percentile'
    VARIATION = 'variation (stdev/avg)'

    COLUMNS = [AGGREGATED, MIN, MAX, PERCENTILE, VARIATION]


EventInfoDataFrame = NewType('EventInfoDataFrame', pd.DataFrame)


class EventInfoDataFrameColumns:
    NAME = 'name'
    DEVICE = 'device'


@dataclass(frozen=True)
class MetricDefinition:
    """
    Metric definition including all its attributes (name, description, formula, etc...).

    """
    # "current" name of the metric, i.e. the EDP metric name
    name: str

    # corresponding metric name for "per transaction" metrics
    throughput_metric_name: str

    # metric description for documentation purposes
    description: str

    # a human readable version of formula. For documentation purposes
    human_readable_expression: str

    # the metric formula
    formula: str

    # maps event aliases (e.g. "a") to their respective event names (e.g. "INST_RETIRED.ANY")
    event_aliases: Dict[str, str] = field(default_factory=dict())

    # maps constant aliases (e.g. "a") to their value (e.g. "2", "system.sockets[0][0].size",
    # "$samplingTime", $processed_samples)
    constants: Dict[str, str] = field(default_factory=dict())

    # maps retire latency aliases (e.g. "a") to their respective latency names
    retire_latencies: Dict[str, str] = field(default_factory=dict())

    # the "standard" name of the metric
    canonical_name: str = None
