# -*- coding: utf-8 -*-
# Create Date: 2024/10/14
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/types.py
# Description: 定义各种中间类

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any, Union, TypeVar
from typing import TypedDict, Callable, Awaitable
from openai.types.chat import ChatCompletionToolParam
from typing_extensions import Required

if TYPE_CHECKING:
    from .agent import Agent

T = TypeVar('T')


class Tool(TypedDict, total=False):
    tool: Required[ChatCompletionToolParam]
    function: Required[Callable | Awaitable]
    context_variables_parameter_name: str
    context_agent_parameter_name: str


class ContextVariables:

    def __init__(self, initial_vars: dict = None) -> None:

        if initial_vars:
            self._vars = initial_vars
        else:
            self._vars = {}

    def __getitem__(self, key: Any) -> Any:
        return self._vars[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._vars[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._vars[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._vars

    def __repr__(self):
        return f"ContextVariables({self._vars})"

    def update(self, other: Union[dict, 'ContextVariables']):
        if isinstance(other, dict):
            self._vars.update(other)
        else:
            self._vars.update(other._vars)
            
    def get(self, key: Any, default: Any = None) -> Any:
        return self._vars.get(key, default)


@dataclass
class Result:
    agent: Optional['Agent'] = None
    content: str = 'Function call successfully.'
    context_variables: ContextVariables | dict = ContextVariables()
    message: bool = True

    def __repr__(self):
        return str({
            'agent': self.agent.name if self.agent else '',
            'content': self.content,
            'context_variables': self.context_variables,
            'message': self.message
        })
