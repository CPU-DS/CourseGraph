# -*- coding: utf-8 -*-
# Create Date: 2024/10/14
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/agent.py
# Description: 定义智能体

from ..llm import LLM
from .tool import Tool
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageParam
import inspect
import docstring_parser
from typing import Callable
from dataclasses import dataclass
import copy
import json


class Agent:

    def __init__(self,
                 name: str,
                 llm: LLM,
                 functions: list[Callable] | None = None,
                 instruction: str = 'You are a helpful assistant.') -> None:
        """ 智能体类

        Args:
            name (str): 名称
            llm (LLM): 大模型
            functions: (list[Callable], optional): 工具函数. Defaults to None.
            instruction (str, optional): 指令. Defaults to 'You are a helpful assistant.'.
        """
        self.name = name
        self.llm = llm
        self.llm.instruction = instruction

        self.tools: list[ChatCompletionToolParam] = []
        self.tool_functions: dict[str, Callable] = {}

        if functions:
            self.add_tool_functions(*functions)

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        """ 历史记录
        """
        return self.llm.messages

    def add_messages_tool_call(self, tool_content: str,
                               tool_call_id: str) -> None:
        """ 向历史记录中添加工具调用的历史记录

        Args:
            tool_content (str): 工具调用结果
            tool_call_id (str): 工具id
        """
        self.llm.messages.append({
            "role": "tool",
            "content": tool_content,
            "tool_call_id": tool_call_id
        })

    def add_tools(self, *tools: 'Tool') -> 'Agent':
        """ 添加外部工具

        Args:
            *tools (Tool): 外部工具

        Raises:
            ValueError: 传递 lambda 函数

        Returns:
            Agent: 智能体
        """
        for tool in tools:
            self.tools.append(tool["tool"])
            function = tool["function"]
            function_name = tool.get('function_name', function.__name__)
            if function_name == '<lambda>':
                raise ValueError(f"不支持 lambda 函数传递")
            self.tool_functions[function_name] = function
        return self

    def add_tool_functions(self, *functions: Callable) -> 'Agent':
        """ 添加外部工具函数，并从函数文档中解析函数描述、参数类型、以及参数描述等信息 \n
        支持的的风格: ReST, Google, Numpydoc-style and Epydoc \n
        参考: https://github.com/rr-/docstring_parser \n
        若使用了不支持的风格且需要函数描述等支持，或者使用复杂参数，类型请使用 add_tools 方法手动编写

        Args:
            *functions (Callable): 外部工具函数

        Raises:
            ValueError: 传递 lambda 函数

        Returns:
            Agent: 智能体
        """

        type_map = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object',
            'None': 'null',
        }

        for function in functions:
            function_name = function.__name__
            if function_name == '<lambda>':
                raise ValueError(f"不支持 lambda 函数传递")
            docstring = inspect.getdoc(function)
            res = docstring_parser.parse(docstring)

            description = (res.short_description or '') + (
                '\n' + res.long_description if res.long_description else '')

            properties = {}
            required = []
            param_names = list(inspect.signature(function).parameters.keys())

            for name in param_names:
                # 参数名称从函数签名获取
                for doc_param in res.params:
                    if doc_param.arg_name == name:
                        properties[name] = {
                            **({
                                'type': type_map[doc_param.type_name]  # 出现不支持的类型会报错
                            } if doc_param.type_name is not None else {}),
                            **({
                                'description': doc_param.description
                            } if doc_param.description is not None else {}),
                        }
                        if not doc_param.is_optional:
                            required.append(name)
                        break
                else:
                    # 无参数描述
                    properties[name] = {}

            self.add_tools({
                'function': function,
                'tool': {
                    'type': 'function',
                    'function': {
                        'name': function_name,
                        'description': description,
                        'parameters': {
                            'type': 'object',
                            'properties': properties,
                            'required': required
                        } if len(properties) != 0 else {}
                    },
                }
            })
        return self


@dataclass
class Response:
    agent: Agent
    content: str


def run(agent: Agent, message: str) -> Response:
    """ 运行 Agent

    Args:
        agent: 智能体
        message (str): 用户输入

    Returns:
        str: Agent 最终输出
    """
    activate_agent = agent
    assistant_output = activate_agent.llm.chat_with_messages(
        message,
        content_only=False,
        tools=activate_agent.tools,
        name=activate_agent.name)
    while assistant_output.tool_calls:  # None 或者空数组
        functions = assistant_output.tool_calls
        for item in functions:
            function = item.function
            tool_function = activate_agent.tool_functions.get(function.name)
            if tool_function:
                tool_content = tool_function(**eval(function.arguments))
                match tool_content:
                    case Agent() as new_agent:
                        if agent.llm is not new_agent.llm:
                            new_agent.messages = copy.deepcopy(
                                new_agent.messages)
                        activate_agent = new_agent
                        activate_agent.add_messages_tool_call(
                            json.dumps({"assistant": activate_agent.name}),
                            item.id)
                    case str():
                        activate_agent.add_messages_tool_call(
                            tool_content, item.id)
                    case _:
                        activate_agent.add_messages_tool_call(
                            "Function call successful.", item.id)
        assistant_output = activate_agent.llm.chat_with_messages(
            content_only=False,
            tools=activate_agent.tools,
            name=activate_agent.name)
    return Response(agent=activate_agent, content=assistant_output.content)
