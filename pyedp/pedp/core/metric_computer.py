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
import logging
import types
from typing import Dict, Callable, List

import numpy as np
import pandas as pd

from pedp.core.internals.code_generator import generate_numpy_vectorized_code
from pedp.core.internals.code_generator import is_number
from pedp.core.types import MetricDefinition


class CompiledMetric:
    """
    A compiled metric that can be executed as a regular python function
    """

    def __init__(self, metric_def: MetricDefinition,
                 compiled_metric_func: Callable,
                 metric_func_source_code: str = None):
        self.__metric_def = metric_def
        self.__metric_function: Callable = compiled_metric_func
        self.__metric_source_code = metric_func_source_code

    def __call__(self, *args, **kwargs):
        """
        Execute the compiled metrics
        :param args: additional arguments
        :param kwargs: additional optional/keyword arguments
        :return: computed metric
        """
        return self.__metric_function(*args, **kwargs)


    @property
    def definition(self) -> MetricDefinition:
        """
        :return: metric definition
        """
        return self.__metric_def

    @property
    def source_code(self) -> str:
        """
        :return: metric function source code
        """
        return self.__metric_source_code


class MetricComputer:
    """
    Compute metrics on a given dataframe
    """
    __module_index = 0

    GROUP_INDEX_NAME = 'group'
    TIMESTAMP_INDEX_NAME = 'timestamp'

    def __init__(self, metric_definition_list: List[MetricDefinition], symbol_table: Dict = {}):
        """
        Initialize the metric computer

        :param metric_definition_list: List of metrics to compute
        :param symbol_table: Dictionary that maps constant names, referenced in metric formulas,
                             to their values
        """
        self.__group_index_name = None
        self.__timestamp_index_name = None
        self.__metric_definition_list = metric_definition_list
        self.__symbol_table = symbol_table
        self.__update_symbol_table(metric_definition_list)
        self.__block_level_metrics_requested = False
        MetricComputer.__module_index += 1
        self.__compiled_metrics = self.__generate_compiled_metrics(metric_definition_list, self.__symbol_table)

    @property
    def symbol_table(self):
        """
        :return: a copy of the symbol table dictionary with which the `MetricComputer` was initialized
        """
        return self.__symbol_table.copy()

    @property
    def compiled_metrics(self) -> List[CompiledMetric]:
        """
        :return: a list of compiled metrics
        """
        return self.__compiled_metrics.copy()

    def compute_metric(self, df: pd.DataFrame,
                       constant_values: Dict = None,
                       calculate_block_level: bool = False,
                       group_index_name: str = GROUP_INDEX_NAME,
                       timestamp_index_name: str = TIMESTAMP_INDEX_NAME) -> pd.DataFrame:
        """
        Compute metrics on the input dataframe

        :param df: input dataframe containing event counts.
                   `df` is expected to have the following structure:
                   - Columns: event names, where each column represents a single event
                   - Index: If `calculate_block_level` is True, `df` must have a multi-index where the first two levels
                            represent timestamp and event group id.
                            `df` can have additional levels, e.g. for socket, core, thread...
                   - Rows: event counts for each event
        :param constant_values: an optional dictionary that maps constant expressions, used in metric formulas,
                                to their values (e.g. 'system.socket_count')
        :param calculate_block_level: whether to calculate block level metrics in addition to sample level metrics
        :param group_index_name: index level name in `df` to use for determining event group id.
                                 Used only when `calculate_block_level` is True.
        :param timestamp_index_name: index level name in `df` to use for determining the timestamp.
                                     Used only when `calculate_block_level` is True.

        :return: a new dataframe where each column is a metric. The index of the result dataframe is identical to `df`
        """
        def verify_preconditions():
            if calculate_block_level:
                expected_index_names = df.index.names[:2]
                if group_index_name not in expected_index_names or timestamp_index_name not in expected_index_names:
                    raise KeyError(f'Unable to calculate block-level metrics: '
                                   f'"{group_index_name}" and "{timestamp_index_name}" must be in the input '
                                   f'dataframe index')

        verify_preconditions()
        self.__group_index_name = group_index_name
        self.__timestamp_index_name = timestamp_index_name
        self.__block_level_metrics_requested = calculate_block_level
        df_with_constants = self.__add_constant_values_to_input_dataframe(df, constant_values)
        sample_level_metrics_df = self.__calculate_sample_level_metrics(df_with_constants, self.compiled_metrics)
        block_level_metrics_df = self.__calculate_block_level_metrics(
                df_with_constants, sample_level_metrics_df, self.compiled_metrics)
        all_metrics_df = self.__merge_sample_and_block_results(sample_level_metrics_df, block_level_metrics_df)

        return all_metrics_df

    @staticmethod
    def __merge_sample_and_block_results(sample_level_metrics_df, block_level_metrics_df):
        if block_level_metrics_df.empty:
            return sample_level_metrics_df

        sample_level_metrics_df.update(block_level_metrics_df)
        return sample_level_metrics_df

    @staticmethod
    def __import_code(code, module_name):
        module = types.ModuleType(module_name)
        exec(code, module.__dict__)
        return module

    def __generate_compiled_metrics(self, metric_definition_list, symbol_table):
        vectorized_compute_metric_code = _MetricCompiler().compile(metric_definition_list, symbol_table)
        generated_module_name = f'generated_code_{self.__module_index}'
        generated_module = self.__import_code(vectorized_compute_metric_code, generated_module_name)
        metric_mapping = generated_module.get_metrics_mapping()

        compiled_metrics = [CompiledMetric(metric, metric_mapping[metric.name][0], metric_mapping[metric.name][1])
                            for metric in self.__metric_definition_list]

        return compiled_metrics

    @staticmethod
    def __all_metric_references_are_available(metric_def: MetricDefinition,
                                              df: pd.DataFrame,
                                              system_symbols: Dict) -> bool:
        for event in metric_def.event_aliases:
            symbol_name = metric_def.event_aliases[event]
            if symbol_name not in df.columns:
                logging.debug(f'Excluding {metric_def.name} from reports because {symbol_name} is unavailable.')
                return False

        for constant in metric_def.constants:
            symbol_name = metric_def.constants[constant]
            if not is_number(symbol_name) and \
                    symbol_name not in system_symbols and \
                    symbol_name not in df.columns:
                logging.debug(f'Excluding {metric_def.name} from reports because {symbol_name} is \
                                missing or invalid.')
                return False

        for retire_latency in metric_def.retire_latencies:
            symbol_name = metric_def.retire_latencies[retire_latency]
            if symbol_name not in df.columns:
                logging.debug(f'Excluding {metric_def.name} from reports because {symbol_name} is unavailable.')
                return False

        return True

    def __calculate_sample_level_metrics(self,
                                         df: pd.DataFrame,
                                         metrics_to_compute: List[CompiledMetric]) -> pd.DataFrame:
        """
        Compute sample-level metrics for the input dataframe.
        Sample level metrics are only calculated for rows in the input dataframe that contain all values
        references by the metric formula.

        :param df: input dataframe
        :param metrics_to_compute: list of metrics to compute

        :return: a dataframe with calculated sample-level metrics, and an index identical to `df`.
        """
        result_df = pd.DataFrame()
        result = {compiled_metric.definition.name: np.array(compiled_metric(df))
                  for compiled_metric in metrics_to_compute
                  if self.__all_metric_references_are_available(compiled_metric.definition,
                                                                df, self.__symbol_table)}

        if result:
            result_df = pd.DataFrame(result, index=df.index)

        return result_df

    def __calculate_block_level_metrics(self,
                                        df: pd.DataFrame,
                                        sample_level_result: pd.DataFrame,
                                        metrics_to_compute: List[CompiledMetric]) -> pd.DataFrame:
        """
        Compute block-level metrics for the input dataframe.

        Block-level metrics are metrics that require events across more than one sample within a block.
        For these metrics, we calculate the block average for all of the events, then use these averages to calculate
        the metric values.
        We then apply the block average metric values to all metrics that weren't calculated per sample,
        and store this value in the last timestamp of the block. This allows the metric to be averaged with all
        the other events/metrics into larger intervals without distorting the data.

        :param df: input dataframe
        :param sample_level_result: dataframe containing the computed sample-level metrics, indexed by timestamp
        :param metrics_to_compute: list of metrics to compute

        :return: a dataframe with calculated block-level metrics, indexed by the same levels as the index of `df`.
        """
        if not self.__block_level_metrics_requested:
            return pd.DataFrame()

        block_avg_df = self.__compute_event_avg_per_group(df)

        # Compute metrics
        block_result_df = pd.DataFrame()
        block_result = {compiled_metric.definition.name: np.array(compiled_metric(block_avg_df)).flatten()
                        for compiled_metric in metrics_to_compute
                        if self.__all_metric_references_are_available(compiled_metric.definition,
                                                                      df, self.__symbol_table)
                        and sample_level_result[compiled_metric.definition.name].isnull().values.all()}

        if block_result:
            block_result_df = pd.DataFrame(block_result, block_avg_df.index)

        return block_result_df

    def __compute_event_avg_per_group(self, df):
        # Compute event average values for each device (socket, core, thread) in each group
        index_names = list(df.index.names)
        index_names.remove(self.__timestamp_index_name)
        block_avg_df = df.groupby(index_names).mean(numeric_only=True)

        # Associate the event averages with the group's max timestamp.
        # This is done by recreating the index, adding each group's max timestamp to the index,
        # and reusing the original index levels order
        block_avg_df.index = pd.MultiIndex.from_frame(
            df.index.to_frame()[self.__timestamp_index_name].groupby(
                index_names).max().reset_index()[df.index.names])
        return block_avg_df

    def __update_symbol_table(self, metric_definition_list: List[MetricDefinition]):
        self.__update_symbol_table_constants()
        self.__update_symbol_table_metric_constants(metric_definition_list)

    def __update_symbol_table_constants(self):
        updated_symbol_table_constants = self.__update_system_constants(self.__symbol_table)
        self.__symbol_table.update(updated_symbol_table_constants)

    def __update_symbol_table_metric_constants(self, metric_definition_list: List[MetricDefinition]):
        for metric in metric_definition_list:
            constants = self.__update_system_constants(metric.constants)
            self.__symbol_table.update(constants)

    @staticmethod
    def __update_system_constants(constants: Dict[str, str]):
        """
        Special logic to extract the value out of 'per_socket' metric constants so these can be set to 1 for uncore
        views. Any special logic to change the symbol_table that requires the metric_constants should go here.

        This only needs to be done if there is a hardcoded metric constant in the metric file that needs to be updated
        later for a specific view.
        i.e. {'channels_populated_per_socket': 8} becomes {'channels_populated_per_socket':
        'system.channels_populated_per_socket', 'system.channels_populated_per_socket': 8}
        @param constants: a dictionary of a metric constant name and corresponding value
        @return: An updated metric constant dictionary for (previously) hardcoded metric constants
        """
        updated_system_constants = {}
        for name, value in constants.items():
            if str(value).isnumeric() and name.endswith('per_socket') and 'system.' not in name:
                system_constant = f'system.{name}'
                updated_system_constants[name] = system_constant
                updated_system_constants[system_constant] = float(value)
        return updated_system_constants

    @staticmethod
    def __add_constant_values_to_input_dataframe(df: pd.DataFrame, constant_values: dict):
        if not constant_values:
            return df
        df_with_constants = df.copy()
        for constant_name, constant_value in constant_values.items():
            df_with_constants[constant_name] = constant_value
        return df_with_constants


class _MetricCompiler:
    """
    Generate executable Python functions from `MetricDefinition` objects
    """
    __vectorized_compute_metric_code_prefix = (
        'import numpy as np\n'
        'import pandas as pd\n'
        'from warnings import simplefilter\n\n\n'
        'def get_metrics_mapping():\n'
        '    np.seterr(all=\'ignore\')\n'
        '    simplefilter(action=\'ignore\', category=pd.errors.PerformanceWarning)\n'
        '    metric_mapping = {\n'
        '        ')

    __vectorized_compute_metric_code_suffix = (
        '\n'
        '        }\n'
        '    return metric_mapping\n'
    )

    def compile(self, metrics: List[MetricDefinition], symbol_table=None) -> str:
        """
        Create a new module and write the vectorized code for computation
        of each metric in the input metric list in a file 'computer_metric.py' inside the module.
        The module is named 'generated_code_x' where x represents the number of MetricComputer instances created.

        :param metrics: List[MetricDefinition] List of input metrics for which vectorized code will be generated
        :param symbol_table: Dictionary that maps constant names, referenced in metric formulas,
                             to their values

        :return module_name.compute_metric: name of the module that needs to imported in order
              to use the generated code
        """
        if symbol_table is None:
            symbol_table = {}
        inlined_system_constants = self.__filter_symbols_to_inline(symbol_table)

        metric_functions = []
        for metric in metrics:
            metric_source_code = self.__create_metric_function(metric, inlined_system_constants)
            metric_functions.append(f'"{metric.name}": '
                                    f'({metric_source_code}, "{metric_source_code}"),')
        vectorized_compute_metric_code_body = '\n        '.join(metric_functions)
        vectorized_compute_metric_code = f'{self.__vectorized_compute_metric_code_prefix}' \
                                         f'{vectorized_compute_metric_code_body}' \
                                         f'{self.__vectorized_compute_metric_code_suffix}'

        return vectorized_compute_metric_code

    @staticmethod
    def _resolve_metric_namespace(metric_namespace: Dict, system_constants: Dict, retire_latencies: Dict) -> Dict:
        """
        Resolves constant references in metric namespace

        :param metric_namespace: a dict representing the metric definition's constants and aliases
        :param system_constants: a dict that maps system constants (e.g., number of sockets) to their values

        :return: a combined dictionary that maps all names referenced by the metric definition to their values
        """
        def adjust_type(v):
            return int(v) if v.isnumeric() else str(v)

        if not system_constants and not retire_latencies:
            return metric_namespace

        resolved_metric_namespace = {}
        constant_aliases = {key: value for key, value in metric_namespace.items() if key != 'metric_name'}
        for key, value in constant_aliases.items():
            if key in retire_latencies and value in system_constants.keys():
                resolved_metric_namespace[key] = system_constants[retire_latencies[key]]
            elif value in system_constants.keys():
                resolved_metric_namespace[key] = system_constants[value]
            elif key in system_constants.keys():
                resolved_metric_namespace[key] = system_constants[key]
            else:
                resolved_metric_namespace[key] = adjust_type(value)
        resolved_metric_namespace.update(system_constants)
        return resolved_metric_namespace

    def __create_metric_function(self, metric: MetricDefinition, system_constants: Dict) -> str:
        """
        Generate the vectorized metric function for input metric.

        :param metric: metric definition
        :param system_constants: a dict that maps system constants (e.g., number of sockets) to their values.
                                 The values of all constants in this dict will be inlined into the generated code.

        :return: a string corresponding to vectorized metric computation with system constant alias
                 replaced with corresponding values
        """
        # Create function namespace
        namespace = self.__create_metric_namespace(metric)
        resolved_namespace = self._resolve_metric_namespace(namespace, system_constants, metric.retire_latencies)
        function_body = generate_numpy_vectorized_code(metric.formula, resolved_namespace)
        metric_function = f'lambda df: {function_body}'
        return metric_function

    def __create_metric_namespace(self, metric_definition: MetricDefinition) -> dict:
        namespace = {'metric_name': metric_definition.name}
        namespace.update(metric_definition.event_aliases)
        namespace.update(metric_definition.constants)
        namespace.update(metric_definition.retire_latencies)
        return namespace

    @staticmethod
    def __filter_symbols_to_inline(symbol_table):
        # Some symbol values should not be inlined into the compiled functions,
        # so that clients can override their values when computing metrics. Filter them out.
        symbols_that_should_not_be_inlined = ['system.socket_count']
        for symbol in symbol_table:
            if 'per_socket' in symbol and 'system.' in symbol: # exclude 'per_socket' symbols from being inlined
                symbols_that_should_not_be_inlined.append(symbol)
        return dict(filter(lambda item: item[0] not in symbols_that_should_not_be_inlined, symbol_table.items()))
