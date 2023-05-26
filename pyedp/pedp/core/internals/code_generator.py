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

import ast
from typing import Dict


def generate_numpy_vectorized_code(exp_in: str, symbol_table: Dict = None) -> str:
    """
    Generates the numpy vectorized code for the passed in expression
    using the symbol_table if defined.
    :param exp_in: expression for which to generate vectorized code
    :param symbol_table: dictionary that map symbol names to their values
    :return: generated Python code as string
    """
    symbol_table = symbol_table if symbol_table else {}
    try:
        exp_in = exp_in.strip()
        exp = ast.parse(exp_in, mode='eval')
        return _ast_gen(exp.body, symbol_table)
    except Exception as e:
        raise SyntaxError(e)


def is_number(value) -> bool:
    """
    Tests if a value is a number
    :param value: value to test
    :return: True if `value` is an integer or floating point number, False otherwise
    """
    try:
        if type(value) is str:
            value = value.replace(',', '')
        float(value)
        return True
    except ValueError:
        return False


class _NameVisitor(ast.NodeVisitor):
    def __init__(self, symbol_table):
        self.__names = set()
        self.__symbol_table = symbol_table

    def visit_Name(self, node):
        self.__names.add(_ast_gen(node, self.__symbol_table))

    @property
    def names(self):
        return sorted(self.__names)


def _generate_all_finite_test(node, symbol_table):
    visitor = _NameVisitor(symbol_table)
    visitor.visit(node)
    condition = ' & '.join([f'np.isfinite({name})' for name in visitor.names])
    return condition


class _MinMaxCallable:
    def __init__(self, function_name: str):
        self.__function_name = function_name

    @property
    def name(self):
        return self.__function_name

    def generate_code(self, args, symbol_table):
        self._validate_args(args)
        args = ', '.join([_ast_gen(arg, symbol_table) for arg in args])
        return f'{self.name}( {args} )'

    @staticmethod
    def _validate_args(args):
        # Verify min() and max() invocation syntax. Allowed variants:
        # 1. A single list argument with exactly 2 values, e.g. min([1,2])
        # 2. Two arguments, e.g. min(1,2)
        if len(args) == 1:
            if type(args[0]) is not ast.List or len(args[0].elts) != 2:
                raise SyntaxError(f'List argument of min() and max() functions must have exactly 2 values')
        elif len(args) != 2:
            raise SyntaxError(f'min() and max() functions accept either 2 arguments or a single list argument '
                              f'with 2 values')


class _MinCallable(_MinMaxCallable):
    def __init__(self):
        super().__init__('np.minimum')


class _MaxCallable(_MinMaxCallable):
    def __init__(self):
        super().__init__('np.maximum')


__call_for_func_id = {
    'min': _MinCallable(),
    'max': _MaxCallable(),
}

__operators = {
    ast.Add: ' + ',
    ast.Sub: ' - ',
    ast.Mult: ' * ',
    ast.Div: ' / ',
    ast.FloorDiv: ' // ',
    ast.BitOr: ' | ',
    ast.Lt: ' < ',
    ast.Gt: ' > ',
    ast.LtE: ' <= ',
    ast.GtE: ' >= ',
    ast.Eq: ' == ',
    ast.USub: ' -',
    ast.UAdd: ' +',
}


def __handle_if_expr(node, symbol_table):
    # Convert a ternary expression of the form "a if condition else b" to the following numpy vector expression:
    # np.where(all_operands_are_finite(condition), np.where(condition, a, b), np.nan), which means:
    #     "if all operands of 'condition' are finite numbers (not inf, -inf or NaN), execute the ternary expression,
    #      otherwise, return NaN".
    #
    # The "wrapper" np.where() is used for differentiating between the following cases:
    # 1. "condition" is evaluated to False because it is indeed False
    # 2. "condition" is evaluated to False because one of its operands is not a finite number
    #    (np.where evaluates NaN to False).
    #
    # This technique ensures that the vectorized expression will evaluate to NaN if any of the operands of "condition"
    # are not finite numbers, which is the expected behavior for computing EDP sample-level metrics.
    condition = _ast_gen(node.test.left, symbol_table) + __operators[type(node.test.ops[0])] \
                + _ast_gen(node.test.comparators[0], symbol_table)
    return f'np.where({_generate_all_finite_test(node.test, symbol_table)}, ' \
           f'np.where({condition}, {_ast_gen(node.body, symbol_table)}, {_ast_gen(node.orelse, symbol_table)}), ' \
           f'np.nan ' \
           f')'


def __handle_ruby_call(node, symbol_table):
    return __generate_function_call(node.attr, [node.value], symbol_table)


def __handle_call(node, symbol_table):
    return __generate_function_call(node.func.id, node.args, symbol_table)


def __generate_function_call(function_name, function_args, symbol_table):
    if function_name not in __call_for_func_id:
        raise SyntaxError(f'Unsupported function: {function_name}')

    func = __call_for_func_id[function_name]
    return func.generate_code(function_args, symbol_table)


def __handle_list(node, symbol_table):
    return ', '.join(list(map(lambda x: _ast_gen(x, symbol_table), node.elts)))


def __handle_name(node, symbol_table):
    if node.id in symbol_table:
        constant_val = symbol_table[node.id]
        if is_number(constant_val):
            return str(constant_val)
        else:
            return f'df[\'{constant_val}\']'
    else:
        return str(node.id)


def __handle_binary_operator(node, symbol_table):
    return f'({_ast_gen(node.left, symbol_table)}{__operators[type(node.op)]}{_ast_gen(node.right, symbol_table)})'


def __handle_unary_operator(node, symbol_table):
    return f'{__operators[type(node.op)]}{_ast_gen(node.operand, symbol_table)}'


__formulas = {
    ast.Constant: lambda node, symbol_table: str(node.n),
    ast.Num: lambda node, symbol_table: str(node.n),
    ast.Name: __handle_name,
    ast.List: __handle_list,
    ast.BinOp: __handle_binary_operator,
    ast.UnaryOp: __handle_unary_operator,
    ast.IfExp: __handle_if_expr,
    ast.Call: __handle_call,
    ast.Attribute: __handle_ruby_call
}


def _ast_gen(node, symbol_table) -> str:
    return __formulas[type(node)](node, symbol_table)
