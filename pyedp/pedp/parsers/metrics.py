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
import json
import jsonschema.exceptions
from jsonschema import validate
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

import defusedxml.ElementTree as ET

from pedp.core.types import MetricDefinition
from pedp.core.validators import FileValidator


def validate_file(file_path: Path):
    file_validator = FileValidator(file_must_exist=True, max_file_size=10 * 1024 * 1024)
    file_validator(str(file_path))


class JsonObjectValidator():
    '''
    Validate json file
    '''

    def __init__(self, file_spec: Path, schema: dict):
        """
        :param: json_file: json file to be validated
        :param: json_schema: json schema to validate the json_file against
        """
        self._json_file = file_spec
        self._json_object = None
        validate_file(file_spec)
        self.__validate_json_syntax(file_spec)
        self.__validate_json_schema(schema)

    @property
    def json_object(self):
        return self._json_object

    def __validate_json_syntax(self, file_spec):
        with open(file_spec) as json_file:
            try:
                self._json_object = json.load(json_file)
            except json.decoder.JSONDecodeError as ex:
                raise SyntaxError(f'syntax error in {self._json_file}')

    def __validate_json_schema(self, schema):
        try:
            validate(self._json_object, schema)
        except jsonschema.exceptions.ValidationError:
            raise jsonschema.exceptions.ValidationError(f'{self._json_file} does not conform to schema')


class MetricDefinitionParser(ABC):
    """
    Metric definition parser abstract base class (ABC)
    """

    def __init__(self, file_path: Path):
        """
        Constructor
        :param file_path: metric definition file to parse
        """
        self._metric_file = file_path

    @abstractmethod
    def parse(self) -> List[MetricDefinition]:
        """
        Parse metric definitions
        :return: list of parsed metrics
        """
        validate_file(self._metric_file)


class JsonParser(MetricDefinitionParser):
    """
    Parser for Data Lake metric definition files (JSON)
    """

    def parse(self) -> List[MetricDefinition]:
        """
        Parse metric definitions
        :return: list of parsed metrics
        """
        super().parse()
        with open(self._metric_file) as metrics_file:
            metric_defs_json = json.load(metrics_file)
            metric_defs = []
            for m in metric_defs_json:
                metric_defs.append(self.__create_from_json(m))
            return metric_defs

    @staticmethod
    def _get_dict_for_tag(tag: str, metric_def):
        d = metric_def.get(str)
        return d if d else {}

    @classmethod
    def __create_from_json(cls, metric_def):
        constants = cls._get_dict_for_tag('constants', metric_def)
        latencies = cls._get_dict_for_tag('retire_latency', metric_def)
        return MetricDefinition(metric_def['name'], '', metric_def.get('description', ''), metric_def['expression'],
                                metric_def['formula'], metric_def['aliases'], constants, latencies,
                                metric_def['name2'])


class XmlParser(MetricDefinitionParser):
    """
    Parser for EDP XML metric definition files
    """

    retire_latency_str = 'retire_latency'

    def parse(self) -> List[MetricDefinition]:
        """
        Parse metric definitions
        :return: list of parsed metrics
        """
        super().parse()
        metric_defs = []
        root = ET.parse(self._metric_file, forbid_dtd=True, forbid_entities=True, forbid_external=True).getroot()
        for metric in root.findall('metric'):
            metric_defs.append(self.__create_from_xml(metric))
        return metric_defs

    @classmethod
    def __create_from_xml(cls, metric_def):
        name = metric_def.get('name')
        throughput_name = cls._get_throughput_name(metric_def)
        constants = cls._get_constants(metric_def)
        events, latencies = cls._get_dicts_for_events(metric_def)
        times = cls._get_dict_for_tag('time', metric_def)
        events.update(times)

        original_formula = metric_def.find('formula').text
        constants, original_formula = cls.__adjust_formula(name, constants, original_formula)
        ruby_python_converter = _RubyPythonConverter(original_formula)
        python_formula = ruby_python_converter.convert()
        return MetricDefinition(name, throughput_name, '', '', python_formula, events, constants, latencies, '')

    @staticmethod
    def _get_dicts_for_events(metric_def):
        events = {}
        latencies = {}
        for md in metric_def.findall('event'):
            if XmlParser.retire_latency_str not in md.text:
                events[md.get('alias')] = md.text
            else:
                latencies[md.get('alias')] = md.text
        return events, latencies

    @staticmethod
    def _get_dict_for_tag(tag: str, metric_def):
        d = {}
        for md in metric_def.findall(tag):
            d[md.get('alias')] = md.text
        return d

    @staticmethod
    def _get_throughput_name(metric_def) -> str:
        for tmn in metric_def.findall('throughput-metric-name'):
            # assume only one <throughput-metric-name> elements
            return tmn.text
        return ''

    @staticmethod
    def _get_constants(metric_def) -> Dict[str, str]:
        constants = {}
        constant_alternatives = {'system.cha_count/system.socket_count': 'chas_per_socket'}
        for const in metric_def.findall('constant'):
            # Hack - replace the "system.cha_count/system.socket_count" expression with a "chas_per_socket" constant
            # TODO: cleanup/refactor
            constants[const.get('alias')] = constant_alternatives.get(const.text.strip(), const.text)
        return constants

    @staticmethod
    def __adjust_formula(name: str, constants: Dict, formula: str):
        # Adjusts the formula of sampling time to get the average sampling time per sample in the aggregate
        if name == "metric_EDP EMON Sampling time (seconds)":
            constants['samples'] = '$processed_samples'
            formula += ' / samples'
        return constants, formula


class _RubyPythonConverter:

    def __init__(self, ruby_expression):
        self.ruby_expression = ruby_expression
        self.regex_patterns = {'ternary': r'(.*)\?(.*):(.*)'}

    def convert(self):
        ruby_expression_type = self.__determine_ruby_expression_type()
        if ruby_expression_type == 'ternary':
            self.ruby_expression = self.__convert_ternary_expression()
        return self.ruby_expression

    def __determine_ruby_expression_type(self):
        for key, pattern in self.regex_patterns.items():
            if re.match(pattern, self.ruby_expression):
                return key

    def __convert_ternary_expression(self):
        out_formula = self.ruby_expression
        # Is there a C-style ternary operator?
        pattern = r'(.*)\?(.*):(.*)'
        ternary_pattern_match = re.match(self.regex_patterns['ternary'], out_formula)
        while ternary_pattern_match:
            # Find each subexpression in the formula
            stack = []
            subexpression_index_pairs = []
            for index, char in enumerate(out_formula):
                if char == '(':
                    stack.append(index + 1)
                elif char == ')':
                    subexpression_index_pairs.append((stack.pop(), index - 1))
            subexpression_index_pairs.append((0, len(out_formula) - 1))
            # Find the innermost subexpression containing the ternary expression and transform it
            for subexpression_index_pair in subexpression_index_pairs:
                subexpression = out_formula[subexpression_index_pair[0]: subexpression_index_pair[1] + 1]
                ternary_match = re.match(pattern, subexpression)
                if ternary_match:
                    [cond, val1, val2] = ternary_match.groups()
                    out_formula = out_formula.replace(subexpression, '{0} if {1} else {2}'.format(val1, cond, val2))
                    break
            ternary_pattern_match = re.match(pattern, out_formula)

        return out_formula


class MetricDefinitionParserFactory:
    """
    Create a metric definition parser based on file type. Use `create(file_path)` method to create the appropriate
    metric definition parser.
    """
    parser_for_file_type = {
        '.xml': XmlParser,
        '.json': JsonParser
    }

    @classmethod
    def create(cls, file_path: Path):
        """
        Creates a parser object based on the given file type
        :param file_path: metric file to parse
        :return: an implementation of MetricDefinitionParser suitable for parsing the specified file
        """
        if file_path.suffix not in cls.parser_for_file_type:
            raise ValueError(f'No metric definition parser defined for files of type {file_path.suffix}')
        return cls.parser_for_file_type[file_path.suffix](file_path)


class JsonConstantParser:
    """
    Parser for EDP json retire latency constant definition files
    """

    # retire latency schema for validating -l/--retire-latency json
    # command line input see SDL: T1148/T1149
    # https://sdp-prod.intel.com/bunits/intel/thor-next-gen-edp/thor/tasks/phase/development/11130-T1148/
    # https://sdp-prod.intel.com/bunits/intel/thor-next-gen-edp/thor/tasks/phase/development/11130-T1149/
    schema = {
        "type": "object",
        "properties": {
            "metric:retire_latency": {
                "type": "object",
                "properties": {
                    "MEAN": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1000000.0
                    },
                    "other-stat": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1000000.0
                    },
                },
                "required": ["MEAN"]
            }
        },
    }

    def __init__(self, file_path: Path, constant_descriptor: str):
        """
        :param file_path: file to parse metric constants from
        """
        self.__constant_descriptor = constant_descriptor
        self.__json_file = file_path

    def parse(self) -> Dict[str, float]:
        """
        Parse constants from json file
        :return: dictionary of parsed constants
        """
        json_object_validator = JsonObjectValidator(self.__json_file, JsonConstantParser.schema)
        json_constants = json_object_validator.json_object
        return self._get_constants_for_descriptor(json_constants)

    def _get_constants_for_descriptor(self, json_constants) -> Dict[str, float]:
        """
        Extract constants that match the constant descriptor from a json dictionary
        :param json_constants: dictionary of constants from parsed from json
        :return: dictionary of constants that match the constant descriptor or an
                 empty dictionary if the constant descriptor isn't found
        """
        constants = {}
        for c in json_constants:
            if self.__constant_descriptor in json_constants[c]:
                constants[c] = json_constants[c].get(self.__constant_descriptor)
        return constants

