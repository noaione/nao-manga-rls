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

from abc import ABC, abstractmethod
from functools import reduce
from operator import and_, or_
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, Field

from .dsl import *

__all__ = (
    "AndModel",
    "BaseRuleModel",
    "ConditionModel",
    "NotModel",
    "OrModel",
    "RuleModel",
)


# BaseModel that provides a method to convert to a DSL object
class BaseRuleModel(BaseModel, ABC):
    """Base model for all rule models."""

    @abstractmethod
    def to_dsl(self) -> Rule:
        """Converts this Pydantic model to a DSL object."""
        pass


def validate_condition_operator(operator: str) -> str:
    if is_valid_operator(operator):
        return operator
    raise ValueError(f"Unsupported operator: {operator}")


class ConditionModel(BaseRuleModel):
    """
    Pydantic model for a simple condition.

    This will be converted to a `Condition` DSL object.

    The `value` field can either be a direct value (str, int, etc.)
    or a reference to another context variable using the format "ctx:variable_name".
    """

    op: Literal["condition"] = Field(..., title="Operator")
    """The operator type, always 'condition' for this model."""
    field: str = Field(..., title="Field")
    """The field to evaluate"""
    operator: Annotated[str, AfterValidator(validate_condition_operator)] = Field(..., title="Operator Type")
    """The comparison operator"""
    value: Any = Field(..., title="Value", examples=["ctx:other_variable", 42, "some_string"])
    """The value to compare against, or a context variable reference"""

    def to_dsl(self) -> Condition:
        """
        Converts this Pydantic model to a Condition DSL object.
        It also checks if the 'value' field is a string prefixed with 'ctx:'
        to represent a reference to another context variable.
        """
        final_value = self.value
        # Check if the value is a string representing a context variable reference
        if isinstance(self.value, str) and self.value.startswith("ctx:"):
            # Extract the variable name and convert it to a ValueReference
            variable_name = self.value[4:]  # Slice off "ctx:"
            final_value = ValueReference(variable_name=variable_name)

        return Condition(field=self.field, operator=self.operator, value=final_value)


class AndModel(BaseRuleModel):
    """
    Pydantic model for a logical AND.

    This will be converted to an `And` DSL object.
    """

    op: Literal["and"] = Field(..., title="Operator")
    """The operator type, always 'and' for this model."""
    rules: list[RuleModel] = Field(..., min_length=1, title="Sub-rules")
    """The list of sub-rules to combine with logical AND."""

    def to_dsl(self) -> Rule:
        """Converts this Pydantic model to a chain of And DSL objects."""
        if not self.rules:
            raise ValueError("AND rule requires at least one sub-rule.")
        dsl_rules = [rule.to_dsl() for rule in self.rules]
        # Use reduce to chain the rules with the '&' operator
        return reduce(and_, dsl_rules)


class OrModel(BaseRuleModel):
    """
    Pydantic model for a logical OR.

    This will be converted to a `Or` DSL object.
    """

    op: Literal["or"] = Field(..., title="Operator")
    """The operator type, always 'or' for this model."""
    rules: list[RuleModel] = Field(..., min_length=1, title="Sub-rules")
    """The list of sub-rules to combine with logical OR."""

    def to_dsl(self) -> Rule:
        """Converts this Pydantic model to a chain of Or DSL objects."""
        if not self.rules:
            raise ValueError("OR rule requires at least one sub-rule.")
        dsl_rules = [rule.to_dsl() for rule in self.rules]
        # Use reduce to chain the rules with the '|' operator
        return reduce(or_, dsl_rules)


# Pydantic model for a logical NOT.
class NotModel(BaseRuleModel):
    """
    Pydantic model for a logical NOT.

    This will be converted to a `Not` DSL object.
    """

    op: Literal["not"] = Field(..., title="Operator")
    """The operator type, always 'not' for this model."""
    rule: RuleModel = Field(..., title="Sub-rule")
    """The sub-rule to negate."""

    def to_dsl(self) -> Rule:
        """Converts this Pydantic model to a Not DSL object."""
        return ~self.rule.to_dsl()


RuleModel: TypeAlias = Annotated[
    ConditionModel | AndModel | OrModel | NotModel,
    Field(discriminator="op", title="Rule"),
]
"""
A union type representing any of the supported rule models.
"""
