# -*- coding: utf-8 -*-
# Create Date: 2024/09/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/tool.py
# Description: 提供常用的工具函数

import requests
import json
from typing import TypedDict, Callable
from openai.types.chat import ChatCompletionToolParam
from typing_extensions import Required


class Tool(TypedDict, total=False):
    tool: Required[ChatCompletionToolParam]
    function: Required[Callable]
    function_name: str  # 需要与tool.function.name相同，作为function的索引
    context_variables_parameter_name: str
    context_agent_parameter_name: str
