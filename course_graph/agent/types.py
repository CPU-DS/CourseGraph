# -*- coding: utf-8 -*-
# Create Date: 2024/10/14
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/types.py
# Description: 定义各种中间类

from addict import Dict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class ContextVariables:
    vars: Dict = field(default_factory=Dict)


@dataclass
class Response:
    agent: 'Agent'
    content: str
    context_variables: ContextVariables


@dataclass
class Result:
    agent: Optional['Agent'] = None
    content: str = 'Function call successful.'
    context_variables: ContextVariables = ContextVariables()
