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
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import reduce
from pathlib import Path
from typing import List, Dict, Generator, Any, Optional, TextIO

import numpy as np
import pandas as pd
import pytz

from pedp.core.types import RawEmonDataFrameColumns as redc, RawEmonDataFrame, EventInfoDataFrame
from pedp.parsers.emon_system_information import EmonSystemInformationParser

UNCORE_UNIT_RE = re.compile(r'UNC_(.*?)_')
FREERUN_RE = re.compile(r'FREERUN_(.*?)_')
FREERUN_SCOPED_RE = re.compile(r'FREERUN:.*scope=(.*)')


def get_event_device(event_name: str) -> str:
    """
    Return device name associated with an event

    :param event_name: the event
    :return: the device name associated with the event
    """
    # TODO: add enumerator for event_type (SYSTEM, CORE, UNCORE), core_type can be in 'device'
    # TODO: get core_type with event
    DEVICE_CORE = 'core'
    DEVICE_UNCORE = 'UNCORE'
    DEVICE_PACKAGE = 'PACKAGE'

    match = UNCORE_UNIT_RE.search(event_name)
    if match:
        return 'UNC_'+match.group(1)

    match = FREERUN_RE.search(event_name)
    if match:
        return match.group(1)

    match = FREERUN_SCOPED_RE.search(event_name)
    if match:
        scope = match.group(1)
        if scope == DEVICE_PACKAGE:
            return scope

    if event_name.upper().startswith('UNC_'):
        return DEVICE_UNCORE

    return DEVICE_CORE


@dataclass(frozen=True)
class Partition:
    """
    Represents a section of a performance data file (e.g., EMON.dat)
    """

    # Number of the first sample in the partition (first sample is 1)
    first_sample: int = 0

    # Number of the last sample in the partition
    last_sample: int = 0

    # Number of sample blocks/loops in the partition
    blocks_count: int = 0

    @property
    def total_samples(self):
        """
        The total number of samples in the partition
        """
        return self.last_sample - self.first_sample + 1


_SAMPLE_SEPARATOR = '----------'
_BLOCK_SEPARATOR = '=========='
_EMON_DATE_FORMAT = '%m/%d/%Y %H:%M:%S.%f'
_DATE_PATTERN_RE = re.compile(r'(\d{2})/(\d{2})/(\d{4})\s(\d{2}):(\d{2}):(\d{2}).(\d{3})')


def _is_separator(line) -> bool:
    return line == _SAMPLE_SEPARATOR or line == _BLOCK_SEPARATOR


def _is_block_separator(line) -> bool:
    return line == _BLOCK_SEPARATOR


def _is_timestamp(line: str) -> bool:
    return True if _DATE_PATTERN_RE.match(line) else False


class EmonParser:
    """
    Parse EMON data file (emon.dat)
    """

    def __init__(self, input_file: Path,
                 emon_v_file: Path = None,
                 timezone: pytz.tzinfo = None,
                 ref_tsc_hz: int = 0):
        """
        Initialize the EMON parser

        :param input_file: the EMON data file to parse
        :param emon_v_file: an optional emon-v.dat file containing system information
        :param timezone: an optional timezone object for converting timestamp strings
        :param ref_tsc_hz: an optional system frequency value (in Hz). Overrides system information in the input file
                            (if such information exists)
        """
        self.input_file = input_file
        self.convert_to_datetime = None
        self.__init_time_conversion_function(timezone)
        self.__chunk_iterator = None
        self.system_info = EmonSystemInformationParser(emon_v_file if emon_v_file else input_file, ref_tsc_hz)
        self.__event_info = EventInfoDataFrame(pd.DataFrame())
        self.__very_first_sample = None

    @property
    def event_info(self) -> EventInfoDataFrame:
        """
        :return: a Pandas DataFrame with the following structure:
                     Rows: 1 row for each event
                     Columns:
                     - 'name': event name
                     - 'device': the event's device, e.g. CORE, CHA, ...

        """
        # Return a copy of the event info DataFrame so that callers cannot modify this instance
        return self.__get_events_information()

    @property
    def first_sample_processed(self):
        """
        The first sample number in the last partition processed by the parser.
        """
        if not self.__chunk_iterator:
            raise AttributeError('the "first_sample_processed" attribute is available only after calling the '
                                 '"event_reader" function')
        return self.__chunk_iterator.sample_tracker.first_sample_number_processed

    @property
    def very_first_sample_processed(self):
        """
        The first sample number in the first partition processed by the parser.
        """
        if not self.__chunk_iterator:
            raise AttributeError('the "very_first_sample_processed" attribute is available only after calling the '
                                 '"event_reader" function')
        return self.__very_first_sample

    @property
    def last_sample_processed(self):
        """
        The last sample number in the last partition processed by the parser.
        """
        if not self.__chunk_iterator:
            raise AttributeError('the "last_sample_processed" attribute is available only after calling the '
                                 '"event_reader" function')
        return self.__chunk_iterator.sample_tracker.last_sample_number_processed

    def event_reader(self,
                     from_timestamp: datetime = None,
                     to_timestamp: datetime = None,
                     from_sample: int = None,
                     to_sample: int = None,
                     partition: Partition = None,
                     chunk_size=1) -> Generator[RawEmonDataFrame, None, None]:
        """
        Parse EMON data and return a generator for EMON values.

        :param from_timestamp: include only samples with timestamp equal to or greater than the specified value.
        :param to_timestamp: include only samples with timestamps equal to or less than the specified value.
        :param from_sample: include only samples equal to or greater than the specified sample number
                            (first sample is 1).
        :param to_sample: include only samples equal to or less than the specified sample number.
        :param partition: include only samples from the specified partition. Cannot be combined with any of the
                          `from` and `to` arguments.
                          Use the `EmonParser.partition` function to generate partition objects.
        :param chunk_size: the maximum number of EMON blocks to include in each chunk returned.
                           An EMON block represents all event values collected in a single iteration of all
                           EMON event groups.
                           Setting this parameter to a high value may cause an out of memory error.
                           Setting this parameter to 0 will read the entire file into memory and may cause an
                           out of memory error.
        :return: a generator for EMON event values, represented as a pandas dataframe with the following structure:
                 rows: a single event value
                 columns:
                   TimeStamp: sample timestamp
                   socket: socket/package number (0 based)
                   device: event type/source, e.g. CORE, UNCORE, ...
                   core: core number (0 based)
                   thread: hyper-thread number within the core (0 based)
                   unit: device instance number, e.g. logical core number (0 based)
                   group: the EMON group number in which the event was collected (0 based)
                   name: event name
                   value: event value

        """

        class EventContentHandler(_ContentHandler):
            """
            A content handler for splitting EMON event samples into chunks.
            Chunks are returned as `RawEmonDataFrame` objects.

            See the `_ContentHandler` class for additional information.
            """

            def __init__(self, emon_parser: EmonParser):
                super().__init__()
                self._samples = []
                self._event_buffer = []
                self._emon_parser = emon_parser

            def end_event(self, data: str):
                self._event_buffer.append(data)

            def end_sample(self, sample_number, block_number):
                if self._event_buffer:
                    self._samples.append(_Sample(self._emon_parser, self._event_buffer, block_number))
                    self._event_buffer.clear()

            def get_chunk_data(self) -> Optional[RawEmonDataFrame]:
                if not self._samples:
                    return None

                emon_df = self._create_raw_emon_df(self._samples)
                self._samples.clear()
                return emon_df

            def _create_raw_emon_df(self, samples: List[_Sample]) -> RawEmonDataFrame:
                """
                Returns a pandas DataFrame containing a row for every event, sample, and value within the raw EMON file.

                @param samples: List of Sample objects, containing each individual line from the raw EMON capture.

                @return DataFrame: Contains a row for every event, sample, and value within the raw EMON file.
                """
                dataframe_builder = _EventDataFrameBuilder(self._emon_parser.system_info)
                for sample in samples:
                    if len(sample.events) == 0:
                        continue
                    for event in sample.events.values():
                        dataframe_builder.append_event_values(sample, event)
                    dataframe_builder.append_pseudo_event_values(sample)
                events_df = dataframe_builder.to_dataframe()
                return RawEmonDataFrame(events_df)

        def verify_preconditions():
            if self.system_info.ref_tsc == 0:
                raise ValueError('Unable to determine system frequency. '
                                 'Please provide an EMON system information file (emon-v.dat) or '
                                 'specify a value for the system frequency')

            if len(self.system_info.socket_map) == 0:
                raise ValueError('Unable to determine processor mapping. '
                                 'Please make sure the information is included in the EMON data file, '
                                 'or provide an EMON processor mapping file (emon-m.dat)')

        verify_preconditions()
        content_handler = EventContentHandler(self)
        # Need to set the parser's __very_first_sample to very first sample processed in first partition/chunk
        if partition and not self.__very_first_sample:
            self.__very_first_sample = partition.first_sample
        with open(self.input_file, 'r') as f:
            self.__chunk_iterator = _ChunkIterator(self, f, content_handler, from_timestamp, to_timestamp,
                                                   from_sample, to_sample, partition, chunk_size=chunk_size)
            for chunk in self.__chunk_iterator:
                yield chunk

    def partition(self,
                  from_timestamp: datetime = None,
                  to_timestamp: datetime = None,
                  from_sample: int = None,
                  to_sample: int = None,
                  chunk_size=1) -> List[Partition]:
        """
        Partition the EMON file into consecutive sections. This is useful for processing the file in parallel.

        See `EmonParser.event_reader` for the description of all function arguments.

        :return: a list of partition objects.
        """

        class PartitionContentHandler(_ContentHandler):
            """
            A content handler for splitting EMON event samples into chunks representing partitions.
            Chunks are returned as `Partition` objects.

            See the `_ContentHandler` class for additional information.
            """

            def __init__(self):
                super().__init__()
                self.__first_block_in_partition = None
                self.__first_sample_in_partition = None
                self.__current_sample = self.__first_sample_in_partition
                self.__current_block = self.__first_block_in_partition
                self.__sample_count = 0

            def end_sample(self, sample_number, block_number):
                if self.__first_sample_in_partition is None:
                    self.__first_sample_in_partition = sample_number
                if self.__first_block_in_partition is None:
                    self.__first_block_in_partition = block_number
                self.__sample_count += 1
                self.__current_sample = sample_number

            def end_block(self, block_number: int):
                self.__current_block = block_number

            def get_chunk_data(self) -> Optional[Partition]:
                if self.__first_sample_in_partition > self.__current_sample:
                    return None

                partition = Partition(first_sample=self.__first_sample_in_partition,
                                      last_sample=self.__current_sample,
                                      blocks_count=self.__current_block - self.__first_block_in_partition + 1)
                self.__first_sample_in_partition = self.__current_sample + 1
                self.__first_block_in_partition = self.__current_block + 1
                return partition

        content_handler = PartitionContentHandler()
        with open(self.input_file, 'r') as f:
            chunk_iterator = _ChunkIterator(self, f, content_handler, from_timestamp, to_timestamp,
                                            from_sample, to_sample, chunk_size=chunk_size)
            partitions = [p for p in chunk_iterator]
            return partitions

    def __init_time_conversion_function(self, timezone: pytz.tzinfo):
        conversion_func = {
            True: lambda s: datetime.strptime(s, _EMON_DATE_FORMAT),
            False: lambda s: datetime.fromtimestamp(
                timezone.localize(datetime.strptime(s, _EMON_DATE_FORMAT)).timestamp())
        }
        self.convert_to_datetime = conversion_func[timezone is None]

    def __get_events_information(self) -> EventInfoDataFrame:
        """
        Return a data frame with the following structure:
          Rows: 1 row for each event
          Columns: 'name' (event name), 'device' (the event's device, e.g. CORE, CHA, ...)
        """
        if self.__event_info.empty:
            # We assume the first block in the EMON data file contains all event groups, and thus contain all events
            df = next(self.event_reader(chunk_size=1))
            self.__event_info = df[[redc.NAME, redc.DEVICE]].drop_duplicates().sort_values(by=[redc.NAME, redc.DEVICE],
                                                                                           axis='rows')
        return self.__event_info.copy()


class _Line:
    """
    Store a line from the EMON file, extracting the relevant information as class attributes.

    Extract attributes from a line in the EMON raw file.  Each line represents the data about a given event for
    the sample duration.  Each line may have a breakdown of the event counts of the system across
    sockets/cores/threads/channels/etc.

    """

    def __init__(self, line):
        # EMON lines are tab separated
        """
        Captures the key attributes of an EMON event from a line in the EMON collection file.

        @param line: raw line from the EMON data file
        """
        line_values = line.split('\t')

        # The first element is the statistic name
        self.name = line_values.pop(0)
        self.device = get_event_device(self.name)

        # The second element is the tsc clock count
        self.tsc_count = float(line_values.pop(0).replace(',', ''))

        # The rest of the elements are event counts
        self.values = np.array([v.replace(',', '').replace('N/A', 'nan') for v in line_values], dtype=float)


class _Sample:
    """
    Store a sample from the EMON file, extracting the relevant information as class attributes.

    Extract attributes from a sample in the EMON raw file.  Each line with the same timestamp represents a sample.
    Within a sample there are various events collected.  These events are listed on their own lines and are stored
    in a Line/Event object.

    Each sample starts with a date line.
    """

    def __init__(self, emon_parser, data, block_number):
        self.__emon_parser = emon_parser
        self.events: Dict[str, _Line] = {}
        topdown_events: Dict[str, _Line] = {}
        topdown_events_original_values: Dict[str, np.array] = {}
        self.block_number = block_number
        ref_tsc = self.__emon_parser.system_info.ref_tsc
        self.__normalized_topdown_events = ['PERF_METRICS.RETIRING', 'PERF_METRICS.BAD_SPECULATION',
                                            'PERF_METRICS.FRONTEND_BOUND', 'PERF_METRICS.BACKEND_BOUND']

        # If line starts with date format, we are at the beginning of the block
        for line in data:
            if _DATE_PATTERN_RE.match(line):
                # Found beginning of a block
                self.TimeStamp = self.__emon_parser.convert_to_datetime(line)
            else:
                try:
                    event = _Line(line)
                    self.events[event.name] = event
                    if self.__is_topdown_event(event.name):
                        topdown_events[event.name] = event
                        topdown_events_original_values[event.name] = event.values.copy()
                except Exception as e:
                    pass
                    # Invalid line - ignore and continue
                    # TODO: log error
                    # TODO: ignore certain errors?
                    # print(e)

        if len(self.events) != 0:
            # If stats are empty, retrieve tsc_count for this block (same for all lines in a block)
            # Calculate the duration from the tsc_count
            self.tsc_count = next(iter(self.events.values())).tsc_count
            self.duration = self.tsc_count / ref_tsc
        else:
            # Else set the tsc_count and duration to 0
            self.tsc_count = 0
            self.duration = 0

        # Adjust the values of the PERF_METRICS.* (TMA) events
        # TODO: refactor, clean-up
        perf_metrics = list(filter(lambda e: e.lower().startswith('perf_metrics'), topdown_events.keys()))
        normalized_events = list(filter(lambda e: self.__is_normalizing_topdown_event(e), perf_metrics))

        for topdown_event in perf_metrics:
            topdown_events[topdown_event].values = (topdown_events_original_values[topdown_event] / (
                reduce(np.add, [topdown_events_original_values[e] for e in normalized_events])
            )) * topdown_events_original_values['TOPDOWN.SLOTS:perf_metrics']

    @staticmethod
    def __is_topdown_event(event_name: str) -> bool:
        return event_name.lower().startswith('perf_metrics') or event_name.startswith('TOPDOWN.SLOTS')

    def __is_normalizing_topdown_event(self, event_name: str) -> bool:
        return event_name in self.__normalized_topdown_events


class _SampleTracker:
    """
    Utility class to assist in tracking and filtering EMON samples
    """

    class _FilterMode(Enum):
        TIMESTAMP = auto()
        SAMPLE = auto()

    def __init__(self, from_sample, to_sample, from_timestamp, to_timestamp, partition):
        self.__current_sample = None
        self.__current_sample_number = 0
        self.__current_sample_timestamp = datetime(1970, 1, 1)
        self.__first_sample_number_processed = 0
        self.__is_first_processed_sample_updated = False
        if partition:
            self.__init_sample_range(partition.first_sample, None, partition.last_sample, None)
        else:
            self.__init_sample_range(from_sample, from_timestamp, to_sample, to_timestamp)

    @property
    def from_sample(self):
        return self.__from_sample if self.__mode == self._FilterMode.SAMPLE else self.__from_timestamp

    @property
    def to_sample(self):
        return self.__to_sample if self.__mode == self._FilterMode.SAMPLE else self.__to_timestamp

    @property
    def current_sample(self):
        return self.__current_sample_number if self.__mode == self._FilterMode.SAMPLE \
            else self.__current_sample_timestamp

    @property
    def first_sample_number_processed(self):
        return self.__first_sample_number_processed

    @property
    def last_sample_number_processed(self):
        if self.is_current_sample_greater_than_range_max():
            return self.__current_sample_number - 1
        else:
            return self.__current_sample_number

    def process(self, sample_timestamp: datetime) -> None:
        self.__current_sample_number += 1
        self.__current_sample_timestamp = sample_timestamp
        if not self.__is_first_processed_sample_updated and self.is_current_sample_in_range():
            self.__first_sample_number_processed = self.__current_sample_number
            self.__is_first_processed_sample_updated = True

    def is_current_sample_in_range(self) -> bool:
        return self.from_sample <= self.current_sample <= self.to_sample

    def is_current_sample_greater_than_range_max(self) -> bool:
        return self.current_sample > self.to_sample

    def __init_sample_range(self, from_sample, from_timestamp, to_sample, to_timestamp):
        if from_timestamp or to_timestamp:
            self.__mode = self._FilterMode.TIMESTAMP
            self.__from_timestamp = from_timestamp if from_timestamp else datetime(1970, 1, 1)
            self.__to_timestamp = to_timestamp if to_timestamp else datetime.max
            self.__from_sample = None
            self.__to_sample = None
        else:
            self.__mode = self._FilterMode.SAMPLE
            self.__from_sample = from_sample if from_sample else 1
            self.__to_sample = to_sample if to_sample else sys.maxsize
            self.__from_timestamp = None
            self.__to_timestamp = None


class _ContentHandler:
    """
    Callback interface for the EMON performance data parser.

    The order of events in this interface mirrors the order of the information in the EMON file
    """

    def __init__(self):
        """
        Receive notification of the beginning of the EMON performance data
        (before reading the first event data in the EMON file).

        The parser will invoke this method once, before any other methods in this interface
        """
        pass

    def end_file(self):
        """
        Receive notification of the end of the EMON performance data
        (after reading the last line in the EMON data file).

        The parser will invoke this method once, and it will be the last method invoked during the parse.
        """
        pass

    def end_sample(self, sample_number: int, block_number: int):
        """
        Signals the end of an EMON sample.

        The parser will invoke this method each time it encounters the EMON sample separator (e.g. '----------').

        :param sample_number: the sample number. First sample is 1.
        :param block_number: the block number. First block is 1.
        """
        pass

    def end_block(self, block_number: int):
        """
        Signals the end of an EMON block.

        The parser will invoke this method each time it encounters the EMON block separator (e.g. '==========').

        :param block_number: the block number. First block is 1.
        """
        pass

    def end_event(self, data: str):
        """
        Signals the end of a single EMON event data.

        The parser will invoke this method each time it encounters an EMON event data line.

        @param data: event data.
        """

    def get_chunk_data(self) -> Optional[Any]:
        """
        Signals the end of a chunk, which contains 1 or more blocks.

        The parser will invoke this method after parsing a number of EMON blocks that is less or equal to the
        specified chunk size. The content handler is expected to return the chunk data.

        :return: chunk data or None if there's no data to return.
                 Return type depends on the implementation. Content handlers decide which data to return.
        """
        pass


class _ChunkIterator:
    """
    Iterator that produces chunks of EMON events data. Chunks are aligned to EMON block boundaries.
    """

    def __init__(self,
                 emon_parser: EmonParser,
                 file: TextIO,
                 handler: _ContentHandler,
                 from_timestamp: datetime = None,
                 to_timestamp: datetime = None,
                 from_sample: int = None,
                 to_sample: int = None,
                 partition: Partition = None,
                 chunk_size=1):
        """
        Initialize the chunk iterator

        :param emon_parser: the EMON parser object that owns this chunk iterator.
        :param file: an open file handle of the EMON data file to parse.
        :param handler: a content handler object that implements the EMON performance data parser interface.

        See `EmonParser.event_reader` for the description of all other arguments.
        """

        def verify_preconditions():
            if from_sample and from_timestamp:
                raise ValueError('The "from_sample" and "from_timestamp" arguments are mutually exclusive')
            if to_sample and to_timestamp:
                raise ValueError('The "to_sample" and "to_timestamp" arguments are mutually exclusive')
            if from_sample and to_timestamp:
                raise ValueError('Cannot use both sample numbers and timestamps to specify a sample range')
            if from_timestamp and to_sample:
                raise ValueError('Cannot use both sample numbers and timestamps to specify a sample range')
            if from_timestamp and to_timestamp and from_timestamp > to_timestamp:
                raise ValueError('The specified "from_timestamp" value must be less than or equal to the '
                                 'specified "to_timestamp" value')
            if from_sample is not None and from_sample <= 0:
                raise ValueError('The specified "from_sample" value must be greater than 0')
            if to_sample is not None and to_sample <= 0:
                raise ValueError('The specified "to_sample" value must be greater than 0')
            if from_sample is not None and to_sample is not None and from_sample > to_sample:
                raise ValueError('The specified "from_sample" value must be less than or equal to the '
                                 'specified "to_sample" value')
            if chunk_size < 0:
                raise ValueError('The specified "chunk_size" value must be greater than or equal to 0')

            if partition is not None and (from_sample is not None or from_timestamp is not None or
                                          to_sample is not None or to_timestamp is not None):
                raise ValueError('The "partition" parameter cannot be combined with any of the "from" or "to" '
                                 'parameters')

        verify_preconditions()
        self._next_block_number = 1
        self._file = file
        self._handler = handler
        self._chunk_size = chunk_size
        self._emon_parser = emon_parser
        self.sample_tracker = _SampleTracker(from_sample, to_sample, from_timestamp, to_timestamp, partition)
        self._end_of_file = False
        self._skip_lines_until_first_sample()

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return the next data chunk.
        The type and content of the chunk is determined by the content handler of the iterator.
        """
        if self._end_of_file:
            raise StopIteration()

        for line in self._file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            if _is_timestamp(line):
                sample_timestamp = self._emon_parser.convert_to_datetime(line)
                self.sample_tracker.process(sample_timestamp)
                # NOTE: If there is ever a need to notify content handlers on sample start,
                #       insert the notification here:
                #       `self._handler.start_sample(sample_timestamp)`

            # Find the first sample to process
            if not self.sample_tracker.is_current_sample_in_range():
                if self.sample_tracker.is_current_sample_greater_than_range_max():
                    # We passed the last sample to process, so terminate the iteration
                    # (same as reaching the end of the file)
                    break
                continue
            if _is_separator(line):
                self._handler.end_sample(self.sample_tracker.last_sample_number_processed, self._next_block_number)
                if _is_block_separator(line):
                    current_block_number = self._next_block_number
                    self._next_block_number += 1
                    self._handler.end_block(current_block_number)
                    if self._chunk_size > 0 and (current_block_number % self._chunk_size == 0):
                        return self._handler.get_chunk_data()
            else:
                self._handler.end_event(line)

        # We reach here when we're done processing the file
        self._handler.end_sample(self.sample_tracker.last_sample_number_processed, self._next_block_number)
        self._handler.end_block(self._next_block_number)
        last_chunk = self._handler.get_chunk_data()
        self._handler.end_file()
        self._end_of_file = True
        if last_chunk is not None:
            return last_chunk
        raise StopIteration()

    def _skip_lines_until_first_sample(self):
        for line in self._file:
            line = line.strip()
            if line and line.startswith('Version Info:'):
                break


class _EventDataFrameBuilder:
    """
    Utility class for building a data frame from event values
    """

    def __init__(self, system_info: EmonSystemInformationParser):
        self.system_info = system_info
        self.__data = []

    def append_event_values(self, sample: 'EmonParser._Sample', event: 'EmonParser._Line') -> None:
        unit_count = event.values.size
        core_devices = ['core'] + self.system_info.unique_core_types
        if event.device.lower() in core_devices:
            self._append_core_event(sample, event, unit_count)
        else:
            self._append_noncore_event(sample, event, unit_count)

    def append_pseudo_event_values(self, sample: _Sample) -> None:
        self._append_tsc(sample)
        self._append_sampling_time(sample)
        self._append_processed_samples(sample)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self.__data, columns=redc.COLUMNS)
        return df.dropna(subset=[redc.VALUE])

    def _append_core_event(self, sample: 'EmonParser._Sample', event: 'EmonParser._Line', unit_count: int) -> None:
        # tuple parameter order follows RawEmonDataFrameColumns. All entries must be accounted for
        self.__data.extend(tuple(((sample.TimeStamp, self.system_info.socket_map[processor],
                                   self.system_info.core_type_map[processor],
                                   self.system_info.core_map[processor], self.system_info.thread_map[processor],
                                   processor, self.system_info.module_map[processor],
                                   sample.tsc_count, sample.block_number, event.name,
                                   event.values[index])
                                  for index, processor in enumerate(self.system_info.unique_os_processors))))

    def _append_noncore_event(self, sample: 'EmonParser._Sample', event: 'EmonParser._Line', unit_count: int) -> None:
        socket_count = np.unique(list(self.system_info.socket_map.values())).size
        units_per_socket = unit_count // socket_count
        # tuple parameter order follows RawEmonDataFrameColumns. All entries must be accounted for
        self.__data.extend(tuple(((sample.TimeStamp, index // units_per_socket,
                                   event.device, None, None, index % units_per_socket, None, sample.tsc_count,
                                   sample.block_number, event.name, event.values[index])
                                  for index in range(unit_count))))

    def _append_tsc(self, sample: 'EmonParser._Sample') -> None:
        # Inject the TSC value as a set of TSC core events. Some metric formulas need TSC as a core event
        core_count = len(self.system_info.core_map)
        # tuple parameter order follows RawEmonDataFrameColumns. All entries must be accounted for
        self.__data.extend(tuple(((sample.TimeStamp, self.system_info.socket_map[processor],
                                   self.system_info.core_type_map[processor],
                                   self.system_info.core_map[processor], self.system_info.thread_map[processor],
                                   index, self.system_info.module_map[processor],
                                   sample.tsc_count, sample.block_number,
                                   'TSC', sample.tsc_count)
                                  for index, processor in enumerate(self.system_info.unique_os_processors))))

    def _append_sampling_time(self, sample: 'EmonParser._Sample') -> None:
        # append parameter order follows RawEmonDataFrameColumns. All entries must be accounted for
        self.__data.append((sample.TimeStamp, 0, 'SYSTEM', None, None, 0, None,
                            sample.tsc_count, sample.block_number, '$samplingTime',
                            sample.duration))

    def _append_processed_samples(self, sample: 'EmonParser._Sample') -> None:
        # append parameter order follows RawEmonDataFrameColumns. All entries must be accounted for
        self.__data.append((sample.TimeStamp, 0, 'SYSTEM', None, None, 0, None,
                            sample.tsc_count, sample.block_number, '$processed_samples', 1))
