"""
MIT License

Copyright (c) 2022-present noaione

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, cast

__all__ = (
    "And",
    "Condition",
    "Context",
    "Not",
    "Or",
    "Rule",
    "ValueReference",
    "is_valid_operator",
)


class Context:
    """
    A simple data container class to hold the state for rule evaluation.
    This makes it easy to access context variables like `ctx.variable`.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def set(self, key, value):
        self.__dict__[key] = value


class ValueReference:
    """A simple container to signify that a value is a reference to a context variable."""

    def __init__(self, variable_name: str):
        self.variable_name: str = variable_name

    def __repr__(self) -> str:
        return f"ContextVar({self.variable_name!r})"


class Rule(abc.ABC):
    """
    Abstract Base Class for all rules and logical operators.
    It defines the core `evaluate` method and overloads Python's bitwise
    operators to create a fluent, DSL-like syntax for combining rules.
    """

    @abc.abstractmethod
    def evaluate(self, context: Context) -> bool:
        """
        Evaluates the rule against a given context.

        Args:
            context: A Context object containing the data to check against.

        Returns:
            True if the rule passes, False otherwise.
        """
        pass

    def __and__(self, other: "Rule") -> "And":
        """Overloads the '&' operator to represent a logical AND."""
        return And(self, other)

    def __or__(self, other: "Rule") -> "Or":
        """Overloads the '|' operator to represent a logical OR."""
        return Or(self, other)

    def __invert__(self) -> "Not":
        """Overloads the '~' operator to represent a logical NOT."""
        return Not(self)


class And(Rule):
    """Represents a logical AND operation between two rules."""

    def __init__(self, left: Rule, right: Rule):
        self.left = left
        self.right = right

    def evaluate(self, context: Context) -> bool:
        return self.left.evaluate(context) and self.right.evaluate(context)

    def __repr__(self) -> str:
        return f"({self.left!r} AND {self.right!r})"


class Or(Rule):
    """Represents a logical OR operation between two rules."""

    def __init__(self, left: Rule, right: Rule):
        self.left = left
        self.right = right

    def evaluate(self, context: Context) -> bool:
        return self.left.evaluate(context) or self.right.evaluate(context)

    def __repr__(self) -> str:
        return f"({self.left!r} OR {self.right!r})"


class Not(Rule):
    """Represents a logical NOT operation for a single rule."""

    def __init__(self, rule: Rule):
        self.rule = rule

    def evaluate(self, context: Context) -> bool:
        return not self.rule.evaluate(context)

    def __repr__(self) -> str:
        return f"NOT({self.rule!r})"


def _is_equal(a: Any, b: Any) -> bool:
    return a == b


def _is_not_equal(a: Any, b: Any) -> bool:
    return a != b


def _is_greater(a: Any, b: Any) -> bool:
    return a > b


def _is_less(a: Any, b: Any) -> bool:
    return a < b


def _is_greater_equal(a: Any, b: Any) -> bool:
    return a >= b


def _is_less_equal(a: Any, b: Any) -> bool:
    return a <= b


def _is_in(a: Any, b: Any) -> bool:
    return a in b


def _is_not_in(a: Any, b: Any) -> bool:
    return a not in b


SupportedOperators = {
    "==": _is_equal,
    "is": _is_equal,
    "!=": _is_not_equal,
    "not is": _is_not_equal,
    ">": _is_greater,
    "<": _is_less,
    ">=": _is_greater_equal,
    "<=": _is_less_equal,
    "in": _is_in,
    "not in": _is_not_in,
}


def is_valid_operator(op: str) -> bool:
    """Checks if the given operator is supported."""
    return op.lower() in SupportedOperators


class Condition(Rule):
    """
    A concrete rule that checks a variable in the context against a value.
    This is the basic building block of your conditions.
    """

    def __init__(self, field: str, operator: str, value: Any):
        """
        Initializes a condition.

        Args:
            field: The name of the variable in the context to check.
            operator: The comparison operator (e.g., '==', '!=', '>', '<', 'in').
            value: The value to compare against.
        """
        self.field = field
        self.operator = operator
        self.value = value

        self.op_func = SupportedOperators.get(self.operator.lower())
        if self.op_func is None:
            raise ValueError(f"Unsupported operator: {self.operator}")

    def evaluate(self, context: Context) -> bool:
        """
        Evaluates the condition against the context.
        It safely gets the value from the context and performs the comparison.
        If the 'value' for the condition is a ValueReference, it will be resolved
        from the context as well.
        """
        context_value = context.get(self.field)

        # Resolve the value to compare against. It's either literal or from context.
        comparison_value = self.value
        if isinstance(self.value, ValueReference):
            comparison_value = context.get(self.value.variable_name)

        # If the primary field doesn't exist, the rule typically fails.
        if context_value is None:
            return False

        op_func = cast(Callable[[Any, Any], bool], self.op_func)
        return op_func(context_value, comparison_value)

    def __repr__(self) -> str:
        return f"Condition({self.field} {self.operator} {self.value!r})"
