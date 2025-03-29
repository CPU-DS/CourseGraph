# -*- coding: utf-8 -*-
# Create Date: 2024/09/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/tool.py
# Description: 定义工具类型

from typing import TypedDict, Callable, Awaitable
from openai.types.chat import ChatCompletionToolParam
from typing_extensions import Required


class Tool(TypedDict, total=False):
    tool: Required[ChatCompletionToolParam]
    function: Required[Callable | Awaitable]
    function_name: str  # 需要与tool.function.name相同，作为function的索引
    context_variables_parameter_name: str
    context_agent_parameter_name: str
