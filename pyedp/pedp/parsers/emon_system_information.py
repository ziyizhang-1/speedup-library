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
from functools import reduce
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

PROCESSOR_FEATURES_SECTION = 'Processor Features'
UNCORE_UNITS_SECTION = 'Uncore Performance Monitoring Units'
PROCESSOR_MAP_SECTION = 'OS Processor <'
RDT_SECTION = 'RDT H/W Support'
GPU_SECTION = 'GPU Information'
RAM_FEATURES_SECTION = 'RAM Features'
QPI_FEATURES_SECTION = 'QPI Link Features'
IIO_FEATURES_SECTION = 'IIO Unit Features'


class EmonSystemInformationParser:
    INT_LIKE_RE = re.compile(r'\d+[\d,]*')

    """
    Parse system information stored in EMON data files (emon.dat, emon-v.dat)
    """
    def __init__(self, input_file: Path, ref_tsc_hz: int = 0):
        """
        Initialize the EMON system information parser

        :param input_file: the EMON data file to parse
        :param ref_tsc_hz: an optional system frequency value (in Hz). Overrides system information in the input file
                            (if such information exists)

        """
        self.__input_file = input_file
        self.__socket_map = {}
        self.__core_map = {}
        self.__thread_map = {}
        self.__core_type_map = {}
        self.__unique_core_types = []
        self.__module_map = {}
        self.__has_modules = False
        self.__ref_tsc = 0
        self.__parser_state = None
        self.attributes: Dict[str, Any] = {}
        self.processor_features: Dict[str, Any] = {}
        self.uncore_units: Dict[str, Any] = {}
        self.ram_features: Dict = {}
        self.rdt: Dict = {}
        self.__parse()
        self.__finalize_attributes(ref_tsc_hz)

    @property
    def socket_map(self) -> Dict[int, int]:
        """
        :return: a dict mapping a logical core number to its socket number
        """
        return self.__socket_map

    @property
    def core_map(self):
        """
        :return: a dict mapping a logical core number to its physical core number
        """
        return self.__core_map

    @property
    def thread_map(self):
        """
        :return: a dict mapping a logical core number to its hardware thread number
        """
        return self.__thread_map

    @property
    def core_type_map(self) -> Dict[int, int]:
        """
        :return: a dict mapping an OS Processor number to its core type for hybrid architectures (bigcore or
        smallcore)
        """
        return self.__core_type_map

    @property
    def unique_core_types(self) -> List[str]:
        """
        :return: a list of unique core types (currently [core] or [bigcore, smallcore]
        """
        return self.__unique_core_types

    @property
    def unique_os_processors(self) -> List[int]:
        """
        :return: a list of unique OS Processors, taken from the EMON topology map
        """
        return self.__unique_os_processors

    @property
    def module_map(self):
        """
        :return: a dict mapping a logical core number to its module
        """
        return self.__module_map

    @property
    def has_modules(self) -> bool:
        """
        :return: a bool specifying if the platform has modules
        """
        return self.__has_modules

    @property
    def ref_tsc(self):
        """
        :return: the system frequency in Hz
        """
        return self.__ref_tsc

    def __parse(self):
        self._set_parser_state(EmonSystemInformationParser._DefaultState())
        with open(self.__input_file, 'r') as f:
            try:
                for _, line in enumerate(f):
                    self.__parser_state.process(self, line.strip())
            except StopIteration:
                pass

    def _set_parser_state(self, new_state: '_State'):
        self.__parser_state = new_state

    def _adjust_type(self, value: str):
        try:
            if re.search(self.INT_LIKE_RE, value):
                # value looks like an int. Try to convert
                return int(value.replace(',', ''))

            if value.lower() in ['yes', 'enabled']:
                return True

            if value.lower() in ['no', 'disabled']:
                return False

            # Unable to determine the type of the value, return as is
            return value
        except ValueError:
            return str(value)

    class _State:
        def process(self, context: 'EmonSystemInformationParser', line: str):
            pass

    class _DefaultState(_State):
        DOT_SEPARATED_RE = re.compile(r'^(?P<name>[^/.]+)\.+(?P<value>[^/.][\s\S]+)$')

        def process(self, context: 'EmonSystemInformationParser', line: str):
            if self.__skip_line(line):
                return

            if line.startswith('Version Info:'):
                context._set_parser_state(EmonSystemInformationParser._FinalState())
            elif line.startswith(PROCESSOR_MAP_SECTION):
                context._set_parser_state(EmonSystemInformationParser._ProcessorMappingState())
            elif line.startswith(PROCESSOR_FEATURES_SECTION):
                context._set_parser_state(EmonSystemInformationParser._ProcessorFeaturesState())
            elif line.startswith(UNCORE_UNITS_SECTION):
                context._set_parser_state(EmonSystemInformationParser._UncoreUnitsState())
            elif line.startswith(RDT_SECTION):
                context._set_parser_state(EmonSystemInformationParser._RdtSupportState())
            elif line.startswith(GPU_SECTION):
                context._set_parser_state(EmonSystemInformationParser._GpuInformationState())
            elif line.startswith(RAM_FEATURES_SECTION):
                context._set_parser_state(EmonSystemInformationParser._RamFeaturesState())
            elif line.startswith(QPI_FEATURES_SECTION):
                context._set_parser_state(EmonSystemInformationParser._QpiFeaturesState())
            elif line.startswith(IIO_FEATURES_SECTION):
                context._set_parser_state(EmonSystemInformationParser._IioFeaturesState())
            elif ':' in line:
                # Parse system information with the following format:
                #   key : value
                #
                # Example file section:
                # ...
                # NUMA node(s):                    2
                # NUMA node0 CPU(s):               0-31,64-95
                # NUMA node1 CPU(s):               32-63,96-127
                # ...
                k, v = line.split(':', 1)
                context.attributes[k.strip()] = context._adjust_type(v.strip())
            else:
                # Parse system information with the following format:
                #   key ......... value
                #
                # Example file section:
                # ...
                # Device Type ............... Intel(R) Xeon(R) Processor code named Icelake
                # EMON Database ............. icelake_server
                # Platform type ............. 125
                # ...
                match = self.DOT_SEPARATED_RE.search(line)
                if match:
                    d = match.groupdict()
                    context.attributes[d['name'].strip()] = context._adjust_type(d['value'].strip())

        def __skip_line(self, line: str) -> bool:
            should_skip = line.startswith('Copyright') \
                          or line.startswith('Application Build Date')
            return should_skip

    class _FinalState(_State):
        def process(self, context: 'EmonSystemInformationParser', line: str):
            raise StopIteration()

    class _ProcessorMappingState(_State):
        MAP_TABLE_SEPARATOR = '-----------------------------------------'

        def __init__(self):
            self.__map_values: List[List[str]] = []
            self.__is_table_start = True

        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "Processor Mapping" section and update system attributes

            Example file section: ::
                ...
		    OS Processor <-> Physical/Logical Mapping
		    -----------------------------------------
              OS Processor	  Phys. Package	      Core	Logical Processor	Core Type	Module
            	   0		       0		       0		   0		     bigcore		2
            	   1		       0		       0		   0		     smallcore		0
            	   2		       0		       1		   0		     smallcore		0
            	   3		       0		       2		   0		     smallcore		0
            	   4		       0		       3		   0		     smallcore		0
            	   5		       0		       0		   0		     smallcore		1
            	   6		       0		       1		   0		     smallcore		1
            	   7		       0		       2		   0		     smallcore		1
            	   8		       0		       3		   0		     smallcore		1
            	   9		       0		       0		   1		     bigcore		2
            	   10		       0		       0		   0		     bigcore		3
            	   11		       0		       0		   1		     bigcore		3
            	   12		       0		       0		   0		     bigcore		4
            	   13		       0		       0		   1		     bigcore		4
            	   14		       0		       0		   0		     bigcore		5
            	   15		       0		       0		   1		     bigcore		5
            	   16		       0		       0		   0		     bigcore		6
            	   17		       0		       0		   1		     bigcore		6
            	   18		       0		       0		   0		     bigcore		7
            	   19		       0		       0		   1		     bigcore		7
		    -----------------------------------------
            """
            if line == self.MAP_TABLE_SEPARATOR:
                if self.__is_table_start:
                    self.__is_table_start = False
                else:
                    context._set_processor_maps(self.__map_values)
                    context._set_parser_state(EmonSystemInformationParser._DefaultState())
            else:
                self.__map_values.append(list(filter(lambda s: s != '', [v.strip() for v in line.split('\t')])))

    class _ProcessorFeaturesState(_State):
        BOOL_VAL_RE = re.compile(r'^\s*\((?P<name>[\s\S]+)\)\s+\((?P<value>[\s\S]+)\)$')
        NUMERIC_VAL_RE = re.compile(r'^\s*\((?P<name>[\s\S]+):\s*(?P<value>[\s\S]+)\)$')

        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "Processor Features" section and update system attributes

            Example file section: ::

                ...
                Processor Features:
                    (Thermal Throttling) (Enabled)
                    (Hyper-Threading) (Enabled)
                    (MLC Streamer Prefetching) (Enabled)
                    (MLC Spatial Prefetching) (Enabled)
                    (DCU Streamer Prefetching) (Enabled)
                    (DCU IP Prefetching) (Enabled)
                    (Cores Per Package:   22)
                    (Threads Per Package: 44)
                    (Threads Per Core:    2)
                ...
            """
            if not line:
                # Done with this section
                context._set_parser_state(EmonSystemInformationParser._DefaultState())
                return

            match = re.search(self.BOOL_VAL_RE, line)
            if not match:
                match = re.search(self.NUMERIC_VAL_RE, line)
            if match:
                d = match.groupdict()
                context.processor_features[d['name'].strip()] = context._adjust_type(d['value'].strip())

    class _UncoreUnitsState(_State):
        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "Uncore Performance Monitoring Units" section and update system attributes

            Example file section: ::
                ...
                Uncore Performance Monitoring Units:
                    cha             : 32
                    imc             : 8
                    m2m             : 4
                    qpi             : 3
                    r3qpi           : 3
                    iio             : 6
                    irp             : 6
                    pcu             : 1
                    ubox            : 1
                    m2pcie          : 6
                    rdt             : 1
                ...
            """
            if not line:
                # Done with this section
                context._set_parser_state(EmonSystemInformationParser._DefaultState())
                return

            parts = line.split(':')
            if len(parts) > 1:
                context.uncore_units[parts[0].strip()] = context._adjust_type(parts[1].strip())

    class _RdtSupportState(_State):
        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "RDT H/W Support" section and update system attributes

            Example file section: ::
                ...
                RDT H/W Support:
                    L3 Cache Occupancy		: Yes
                    Total Memory Bandwidth	: Yes
                    Local Memory Bandwidth	: Yes
                    L3 Cache Allocation		: Yes
                    L2 Cache Allocation		: No
                    Highest Available RMID	: 255
                    Sample Multiplier		: 65536
                ...
            """
            if not line:
                # Done with this section
                context._set_parser_state(EmonSystemInformationParser._DefaultState())
                return

            parts = line.split(':')
            if len(parts) > 1:
                context.rdt[parts[0].strip()] = context._adjust_type(parts[1].strip())

    class _GpuInformationState(_State):
        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "GPU Information" section and update system attributes

            Example file section: ::
                ...
                GPU Information:

                TBD...
                ...
            """
            if not line:
                # Done with this section
                context._set_parser_state(EmonSystemInformationParser._DefaultState())
                return

            # TODO: parse GPU Information section

    class _QpiFeaturesState(_State):
        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "QPI Link Features" section and update system attributes

            Example file section: ::
                ...
                QPI Link Features:
                    Package 0 :
                    Package 1 :
                ...
            """
            if not line:
                # Done with this section
                context._set_parser_state(EmonSystemInformationParser._DefaultState())
                return

            # TODO: parse QPI Link Features section

    class _IioFeaturesState(_State):
        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "IIO Unit Features" section and update system attributes

            Example file section: ::
                ...
                IIO Unit Features:
                    Package 0 :
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                    Package 1 :
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                        domain:0 bus:0x00 stack:0 mesh: 0
                ...
            """
            if not line:
                # Done with this section
                context._set_parser_state(EmonSystemInformationParser._DefaultState())
                return

            # TODO: parse IIO Unit Features section

    class _RamFeaturesState(_State):
        DIMM_LOCATION_RE = re.compile(r'\((\d+)/(\d+)/(\d+)\)')
        DIMM_INFO_RE = re.compile(r'\(dimm(?P<id>\d+) info:\s*(?P<value>.*)\)', flags=re.IGNORECASE)

        def __init__(self):
            self.__current_dimm_location = None
            self.__dimm_location_map = {}

        def process(self, context: 'EmonSystemInformationParser', line: str):
            """
            Parses the "RAM Features" section and stores the information in the ram_features attribute

            Example file section: ::
                ...
                RAM Features:
                    (Package/Memory Controller/Channel)
                    (0/0/0) (Total Number of Ranks on this Channel: 2)
                         (Dimm0 Info: Empty)
                         (Dimm1 Info: Empty)
                    (0/0/1) (Total Number of Ranks on this Channel: 2)
                         (Dimm0 Info: Capacity = 32, # of devices = 32, Device Config = 8Gb(2048Mbx4))
                         (Dimm1 Info: Capacity = 32, # of devices = 32, Device Config = 8Gb(2048Mbx4))
                ...
            """
            if not line:
                # Done with this section
                self.__finalize_ram_features(context)
                context._set_parser_state(EmonSystemInformationParser._DefaultState())
                return

            # TODO: initial implementation commented out as it is not reliable
            # if line.startswith('(Package/Memory Controller/Channel'):
            #     return
            #
            # match = re.search(self.DIMM_LOCATION_RE, line)
            # if match:
            #     self.__current_dimm_location = tuple(int(g) for g in match.groups())
            #     return
            #
            # match = re.search(self.DIMM_INFO_RE, line)
            # if match:
            #     # TODO: incomplete implementation. Need to properly parse DIMM attributes
            #     d = match.groupdict()
            #     if not self.__current_dimm_location:
            #         raise ValueError(f'Unable to match memory module specification to its location: "{line}"')
            #     if self.__current_dimm_location not in self.__dimm_location_map:
            #         self.__dimm_location_map[self.__current_dimm_location] = {int(d['id']): d['value']}
            #     else:
            #         self.__dimm_location_map[self.__current_dimm_location].update({int(d['id']): d['value']})

        def __finalize_ram_features(self, context):
            if len(self.__dimm_location_map.keys()) == 0:
                return

            sockets, controlers_per_socket, channels_per_controller = \
                reduce(lambda a, b: (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])),
                       self.__dimm_location_map.keys())
            sockets += 1
            controlers_per_socket += 1
            channels_per_controller += 1

            ram_features = {}
            for socket in range(sockets):
                ram_features[socket] = {}
                for controller in range(controlers_per_socket):
                    ram_features[socket][controller] = {}
                    for channel in range(channels_per_controller):
                        ram_features[socket][controller][channel] = self.__dimm_location_map[(socket, controller,
                                                                                              channel)]
            context.ram_features = ram_features.copy()

    def _set_processor_maps(self, map_values: List[List[str]]):
        df = pd.DataFrame(map_values[1:], columns=map_values[0])
        if 'Core Type' not in df.columns:
            df['Core Type'] = ['core'] * len(df.index)

        if 'Module' not in df.columns:
            df['Module'] = ['0'] * len(df.index)
        else:
            self.__has_modules = True

        try:
            df = pd.concat([df[['OS Processor', 'Phys. Package', 'Core', 'Logical Processor']].astype(int),
                            df['Core Type'], df['Module'].astype(int)], axis=1)
            self.__socket_map = dict(zip(df['OS Processor'].values, df['Phys. Package'].values))
            self.__core_map = dict(zip(df['OS Processor'].values, df['Core'].values))
            self.__thread_map = dict(zip(df['OS Processor'].values, df['Logical Processor'].values))
            self.__core_type_map = dict(zip(df['OS Processor'].values, df['Core Type'].values))
            self.__module_map = dict(zip(df['OS Processor'].values, df['Module'].values))
            self.__unique_core_types = list(df['Core Type'].unique())
            self.__unique_os_processors = list(self.core_map.keys())

        except ValueError:
            # TODO: log an error
            # Swallow exception so that the parser doesn't crash
            pass


    def __finalize_attributes(self, ref_tsc_hz):
        if ref_tsc_hz > 0:
            self.__ref_tsc = ref_tsc_hz
        else:
            self.__set_ref_tsc()

    def __set_ref_tsc(self):
        tsc_candidates = [key for key in self.attributes.keys() if key.startswith('TSC Freq')]
        if len(tsc_candidates) > 0:
            # Assume the first occurrence inside tsc_candidates is the TSC Freq
            ref_tsc = self.attributes[tsc_candidates[0]].split(' ', 1)[0]
        elif 'Processor Base Freq' in self.attributes:
            ref_tsc = self.attributes['Processor Base Freq'].split(' ', 1)[0]
        else:
            # Unable to determine processor frequency
            ref_tsc = 0
        self.__ref_tsc = float(ref_tsc) * 1000000


class EmonSystemInformationAdapter:
    """
    Adapt EMON system information data to `MetricComputer` symbol table format (Dict[str, Any])
    """
    def __init__(self, emon_sys_info: EmonSystemInformationParser):
        self.__emon_sys_info = emon_sys_info

    def get_symbol_table(self) -> Dict[str, Any]:
        """
        :return: a symbol table for `MetricComputer` from EMOM system information
        """
        sockets_count = self.__emon_sys_info.processor_features['Number of Packages']
        symbol_table = {
            'system.tsc_freq': self.__emon_sys_info.ref_tsc,
            'system.sockets[0][0].size': self.__emon_sys_info.processor_features['Threads Per Core'],
            'system.socket_count': sockets_count,
            'system.sockets[0].cores.count': self.__emon_sys_info.processor_features['Cores Per Package'],
            'system.sockets[0].cpus.count': self.__emon_sys_info.processor_features['Threads Per Package'],
        }
        if 'cha' in self.__emon_sys_info.uncore_units:
            symbol_table['chas_per_socket'] = self.__emon_sys_info.uncore_units['cha']
        return symbol_table
