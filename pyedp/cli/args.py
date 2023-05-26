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
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any

from pedp import ViewAggregationLevel
from pedp.core.validators import FileValidator
from pedp.core.types import DeviceType
from cli import __version__

TIMESTAMP_ARG_FORMAT = '%m/%d/%Y %H:%M:%S'

TIMESTAMP_ARG_FORMAT_MS = '%m/%d/%Y %H:%M:%S.%f'

MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


class ValidFileSpec:
    """
    ArgumentParser "type" validator for command line options that accept file names/paths
    """

    def __init__(self, file_must_exist=True, file_must_not_exist=False, max_file_size=0, allow_symlinks=True,
                 check_dir_only=False):
        """
        Constructor

        :param file_must_exist: when True, verifies that the specified file exists
        :param file_must_not_exist: when True, verifies that the specified file does not exist
        :param max_file_size: maximum file size allowed (0 means MAX_FILE_SIZE_BYTES)
        :param allow_symlinks: when True, allow using symlinks for existing files, otherwise reject symlinks
        """
        self.__validate = FileValidator(file_must_exist=file_must_exist,
                                        file_must_not_exist=file_must_not_exist,
                                        max_file_size=max_file_size,
                                        allow_symlinks=allow_symlinks,
                                        check_dir_only=check_dir_only
                                        )

    def __call__(self, file_spec):
        try:
            return self.__validate(file_spec)
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)


class ValidMultiFileSpec(ValidFileSpec):
    """
    ArgumentParser "type" validator for command line options that accept file names/paths and may include
    a prepended core type specifier. ValidMultiFileSpec parses file paths for both single and hybrid core platforms.
    File specification format:
        non-hybrid format: <file name with path>
        hybrid format: <core_type>=<file name with path>
    """

    def __init__(self, file_must_exist=True, file_must_not_exist=False, max_file_size=0, allow_symlinks=True):
        super().__init__(file_must_exist, file_must_not_exist, max_file_size, allow_symlinks)
        self.__core_type: str = DeviceType.CORE  # default core type for non-hybrid platforms
        self.__file: str = ''

    def __call__(self, file_spec: str):
        """
        :param file_spec: A string that represents a file with path. For hybrid files, the string is
                prepended with <core_type>=
                Non-hybrid file format: <file name with path>
                hybrid file format: <core_type>=<file name with path>
        :return: a single element dictionary where key:value is <core_type>:<file name with path>,
                 { <core_type>: <file name with path> }, for non-hybrid entries, the core_type is the
                 default core type.
        """
        self._parse_file_spec(file_spec)
        return {self.__core_type: super().__call__(self.__file)}

    @staticmethod
    def reformat_multi_file_args(file_spec_list: List[Dict[str, str]]):
        """
        For options with a variable number of args, argparse stores a list of objects
        of the defined object type in the argument variable for that option. For variable args of
        type ValidMultiFileSpec, that list is a list of single element dictionaries that include
        exactly one key:value pairing of core_type:file per list element. See the __call__ function
        for this class for more information. To simplify use of the multi file argument for other
        objects/functions that consume args from argparse, convert the list of dictionaries to a single
        dictionary with one entry for each core_type:file pairing.

        :param file_spec_list: A list of single element dictionaries where key:value is core_type:file
        :return: a dictionary of core_type:file pairs

        Example: input -> [{ core_type1: file1 }, { core_type2: file2 }]
                 output <- { core_type1: file1, core_type2: file2 }
        """
        file_args: Dict[str, str] = {}
        for file_spec in file_spec_list:
            if len(file_spec) != 1:
                raise argparse.ArgumentTypeError(f'Invalid file specification: {file_spec}')
            for core_type in file_spec:
                file_args[core_type] = file_spec[core_type]
        return file_args

    @staticmethod
    def validate_multi_file_core_types(file_spec_dict: Dict[str, str], valid_core_types: List[str]):
        """
        This public helper function is provided to check that core types entered on the command line
        are valid. It raises an error if the core_type from the command line arg is not documented in
        the data file unique/valid core types list.
        :param file_spec_dict: The dictionary, produced by the argument parser, for the file specification
                               arg to be validated.
                               (examples: args.metric_file_path and args.chart_format_file_path).
        :param valid_core_types: a list of known valid core types from the data file parser
        """
        for core_type in file_spec_dict:
            if core_type not in valid_core_types:
                raise argparse.ArgumentTypeError(f'{core_type} is not a supported core type')

    def _parse_file_spec(self, file_spec: str):
        """
        Parse the file specification for hybrid and non-hybrid entries
        :param file_spec: a single or hybrid core file specification
        """
        self.__file = file_spec
        hybrid_arg_separator: str = "="
        if hybrid_arg_separator in file_spec:
            args = file_spec.split(hybrid_arg_separator)
            ValidMultiFileSpec._validate_hybrid_args(file_spec, args)
            self.__core_type = args[0]
            self.__file = args[1]
            self._validate_hybrid_core_type()

    @staticmethod
    def _validate_hybrid_args(file_spec: str, args: List[str]):
        """
        Sanity check a hybrid file specification
        It is expected to have exactly two non-empty args
        """
        expected_num_args = 2
        if '' in args or len(args) != expected_num_args:
            raise argparse.ArgumentTypeError(f'Invalid format: {file_spec}. '
                                             f'Expected: <core type>=<file name with path>.')

    def _validate_hybrid_core_type(self):
        """
        Sanity check the hybrid core type
        Hybrid cores cannot be the default core type
        Note: core type can't be fully validated until after the emon system information is parsed in main.
              see validate_multi_file_core_types for validating after both the command line and the data file
              have been parsed.
        """
        if self.__core_type == DeviceType.CORE:
            raise argparse.ArgumentTypeError(f'{self.__core_type} is not a valid hybrid core.')


class ValidSample:
    """
    ArgumentParser "type" validator for command line options that accept sample number or timestamp
    """

    def __call__(self, sample_reference):
        try:
            sample_as_int = int(sample_reference)
            if '.' in str(sample_reference) or sample_as_int < 1:
                raise argparse.ArgumentTypeError(f'{sample_reference}: '
                                                 f'sample number must be equal to or greater than 1')
            return sample_as_int
        except ValueError:
            try:
                return datetime.strptime(sample_reference, TIMESTAMP_ARG_FORMAT_MS)
            except ValueError:
                try:
                    return datetime.strptime(sample_reference, TIMESTAMP_ARG_FORMAT)
                except ValueError:
                    raise argparse.ArgumentTypeError(f'{sample_reference}: not a valid timestamp or sample number')


class IntegerRange:
    """
    ArgumentParser "type" validator for numeric command line options with optional min/max values
    """

    def __init__(self, min_value=0, max_value=sys.maxsize):
        self.__min_value = min_value
        self.__max_value = max_value

    def __call__(self, value):
        if '.' in str(value):
            raise argparse.ArgumentTypeError(f'{value}: value must be an integral (whole) number')

        try:
            value_as_int = int(value)
            if not (self.__min_value <= value_as_int <= self.__max_value):
                raise argparse.ArgumentTypeError(f'{value}: value must be an integral (whole) number between '
                                                 f'{self.__min_value} and {self.__max_value}')
            return value_as_int
        except ValueError:
            raise argparse.ArgumentTypeError(f'{value}: value must be an integral (whole) number')


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    :return: a Namespace object with parsed command line arguments as attributes
    """

    parser = argparse.ArgumentParser(description='EMON Data Processing Tool', add_help=False)

    required_options = parser.add_argument_group('Required options')
    required_options.add_argument('-i', '--input',
                                  dest='emon_dat_file_path',
                                  metavar='emon.dat',
                                  type=ValidFileSpec(file_must_exist=True),
                                  required=True,
                                  help='input file: emon dat',
                                  )
    required_options.add_argument('-m', '--metric',
                                  nargs='+',
                                  dest='metric_file_path',
                                  metavar='metric.xml',
                                  type=ValidMultiFileSpec(file_must_exist=True, max_file_size=MAX_FILE_SIZE_BYTES),
                                  required=True,
                                  help='input file: metric definition file. '
                                       'Specify hybrid files using <hybrid core type>=<metric definition file>',
                                  )
    required_options.add_argument('-o', '--output',
                                  dest='output_file_path',
                                  metavar='output.xlsx or output_dir',
                                  type=ValidFileSpec(file_must_exist=False,
                                                     file_must_not_exist=False,
                                                     allow_symlinks=False,
                                                     check_dir_only=True
                                                     ),
                                  required=True,
                                  help='If an output directory is specified, excel file generation is skipped; '
                                       'otherwise excel file is produced alongside csv files.',
                                  )

    view_options = parser.add_argument_group('View generation options')
    view_options.add_argument('--socket-view',
                              action='store_const',
                              const=ViewAggregationLevel.SOCKET,
                              default=None,
                              help='Generate Socket-level summary and details')
    view_options.add_argument('--core-view',
                              action='store_const',
                              const=ViewAggregationLevel.CORE,
                              default=None,
                              help='Generate Core-level summary and details')
    view_options.add_argument('--thread-view',
                              action='store_const',
                              const=ViewAggregationLevel.THREAD,
                              default=None,
                              help='Generate Thread-level summary and details')
    view_options.add_argument('--uncore-view',
                              action='store_const',
                              const=ViewAggregationLevel.UNCORE,
                              default=None,
                              help='Generate Uncore unit-level summary and details')
    view_options.add_argument('--no-detail-views',
                              action='store_true',
                              default=False,
                              help='Generate only summary views (significantly improves performance)')

    optional_options = parser.add_argument_group('Optional options')
    optional_options.add_argument('-j', '--emonv',
                                  dest='emonv_file_path',
                                  metavar='emon-v.dat',
                                  type=ValidFileSpec(file_must_exist=True, max_file_size=MAX_FILE_SIZE_BYTES),
                                  help='input file: emon -v dat',
                                  )
    optional_options.add_argument('-f', '--format',
                                  nargs='+',
                                  dest='chart_format_file_path',
                                  metavar='format.txt',
                                  type=ValidMultiFileSpec(file_must_exist=True, max_file_size=MAX_FILE_SIZE_BYTES),
                                  help='input file: chart format definition file. '
                                  'Specify hybrid files using <hybrid core type>=<chart format file>',
                                  )
    optional_options.add_argument('-c', '--frequency',
                                  dest='frequency',
                                  type=IntegerRange(min_value=1),
                                  help='TSC frequency in MHz (e.g., -c 1600)',
                                  )
    optional_options.add_argument('-b', '--begin',
                                  dest='begin_sample',
                                  metavar='#SAMPLE or TIMESTAMP',
                                  type=ValidSample(),
                                  help='First sample to process. Specify a sample number or a timestamp '
                                       '(MM/DD/YYYY HH:MM:SS.sss, where sss is milliseconds and is optional)',
                                  )
    optional_options.add_argument('-e', '--end',
                                  dest='end_sample',
                                  metavar='#SAMPLE or TIMESTAMP',
                                  type=ValidSample(),
                                  help='Last sample to process. Specify a sample number or a timestamp '
                                       '(MM/DD/YYYY HH:MM:SS.sss, where sss is milliseconds and is optional)',
                                  )
    optional_options.add_argument('-x', '--tps',
                                  dest='transactions_per_second',
                                  type=IntegerRange(min_value=1),
                                  help='Number of transactions per second for throughput-mode reports',
                                  )
    optional_options.add_argument('-l', '--retire-latency',
                                  dest='retire_latency',
                                  metavar='latency.json',
                                  type=ValidMultiFileSpec(file_must_exist=True, max_file_size=MAX_FILE_SIZE_BYTES),
                                  help='retire latency file (json): retire latency definition file. '
                                       'Specify hybrid files using <hybrid core type>=<retire latency file>',
                                  )
    optional_options.add_argument('--chunk-size',
                                  dest='chunk_size',
                                  type=IntegerRange(),
                                  default=20,
                                  help='Number of EMON "blocks" to process at a time. '
                                       'Higher number requires more memory and may speed up processing. '
                                       'Set to 0 to process the entire input file in memory '
                                       '(may cause an out of memory error when processing large files)',
                                  )

    misc_options = parser.add_argument_group('Miscellaneous options')
    misc_options.add_argument('-h', '--help',
                              help='Show this help message and exit',
                              action='help')
    misc_options.add_argument('--version',
                              help='Show version information and exit',
                              action='version',
                              version=f'%(prog)s {__version__}')
    misc_options.add_argument('--verbose',
                              help='Increase output verbosity',
                              action='store_true')
    return _validate_and_parse_args(parser)


def get_sample_range_args(args: argparse.Namespace) -> Dict[str, Any]:
    if type(args.begin_sample) is datetime or type(args.end_sample) is datetime:
        return {'from_timestamp': args.begin_sample, 'to_timestamp': args.end_sample}
    else:
        return {'from_sample': args.begin_sample, 'to_sample': args.end_sample}


def reformat_multi_file_args(args):
    """
    This function replaces the native argparse arg for multi file options with an
    equivalent, but easier to consume variant of the same information.
    For more information, see the docu-comment for reformat_multi_file_args
    :param args: args object produced by the argparse parser.
    """
    args.metric_file_path = ValidMultiFileSpec.reformat_multi_file_args(args.metric_file_path)
    if args.chart_format_file_path is not None:
        args.chart_format_file_path = ValidMultiFileSpec.reformat_multi_file_args(args.chart_format_file_path)


def _validate_and_parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse and validate command line arguments. Terminate the program if there are errors

    :param parser: the argument parser

    :return: a Namespace object with parsed arguments as attributes
    """
    args = parser.parse_args()
    reformat_multi_file_args(args)
    if args.begin_sample and args.end_sample and type(args.begin_sample) is not type(args.end_sample):
        parser.error('please specify either timestamps or sample numbers for "begin sample" and "end sample"')
    if args.begin_sample and args.end_sample and args.begin_sample > args.end_sample:
        parser.error('the "begin sample" value cannot be greater than the "end sample" value')

    return args
