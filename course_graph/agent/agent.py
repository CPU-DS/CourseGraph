# -*- coding: utf-8 -*-
# Create Date: 2024/10/14
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/agent.py
# Description: 定义智能体

from ..llm import LLM
from .tool import Tool
from .types import Response, Result, ContextVariables
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageParam
import inspect
import docstring_parser
from typing import Callable
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
        self.instruction = instruction

        self.tools: list[ChatCompletionToolParam] = []
        self.tool_functions: dict[str, Callable] = {}
        self.use_context_variables: dict[str, str] = {}  # 使用了上下文变量的函数以及相应的形参名称

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
            if (r := tool.get('context_variables_parameter_name')) is not None:
                self.use_context_variables[function_name] = r
        return self

    def add_tool_functions(self, *functions: Callable) -> 'Agent':
        """ 添加外部工具函数，并从自动解析函数描述、参数类型、以及参数描述等信息 \n
        若使用了不支持的文档风格或使用复杂参数，请调用 add_tools 方法手动编写函数描述

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

            properties: dict[str, dict] = {}
            required: list[str] = []

            signature_parameters = inspect.signature(function).parameters
            # 所有的形参名称
            for name in list(signature_parameters.keys()):
                properties[name] = {}

            for arg_name, p in signature_parameters.items():
                # 移除ContextVariables变量并且忽略默认值
                if p.annotation == ContextVariables:
                    del properties[arg_name]
                    # 保存函数名对应的ContextVariables的形参名称
                    # ContextVariables只允许有一个, 后者会覆盖前者
                    self.use_context_variables[function_name] = arg_name
                    continue
                # 签名中的参数类型
                if p.annotation != inspect._empty:
                    properties[arg_name]['type'] = type_map.get(
                        str(p.annotation).replace("<class '",
                                                  '').replace("'>", ''), 'any')
                # 签名中的默认值
                if p.default == inspect._empty:
                    required.append(arg_name)

            for doc_parameters in res.params:
                if doc_parameters.arg_name in properties:
                    # 文档中的参数类型 (优先使用签名中的参数类型)
                    if doc_parameters.type_name is not None and 'type' not in properties[
                            doc_parameters.arg_name]:
                        properties[
                            doc_parameters.arg_name]['type'] = type_map.get(
                                doc_parameters.type_name, 'any')
                    # 文档中的参数描述
                    if doc_parameters.description is not None:
                        properties[doc_parameters.arg_name][
                            'description'] = doc_parameters.description
                    # 文档中的是否可选 (签名中提示required这里也不会丢掉)
                    if not doc_parameters.is_optional and doc_parameters.arg_name not in required:
                        required.append(doc_parameters.arg_name)

            # 如果签名和文档中都未标注类型则可以使用默认值类型替代
            for arg_name, p in signature_parameters.items():
                if arg_name in properties:
                    if p.default != inspect._empty and 'type' not in properties[
                            arg_name]:
                        properties[arg_name]['type'] = type_map.get(
                            str(type(p.default)).replace("<class '",
                                                         '').replace("'>", ''),
                            'any')

            self.add_tools({  # context_variables_parameter_name 已经处理过无需传递
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


def run(
    agent: Agent,
    message: str,
    context_variables: ContextVariables = ContextVariables()
) -> Response:
    """ 运行 Agent

    Args:
        agent: 智能体
        message (str): 用户输入
        context_variables (str, ContextVariables): 上下文变量. Defaults to ContextVariables().

    Returns:
        str: Agent 最终输出
    """
    activate_agent = agent
    activate_agent.llm.instruction = activate_agent.instruction
    assistant_output = activate_agent.llm.chat_with_messages(
        message,
        content_only=False,
        tools=activate_agent.tools,
        name=activate_agent.name)
    while assistant_output.tool_calls:  # None 或者空数组
        functions = assistant_output.tool_calls
        for item in functions:
            function = item.function
            if (tool_function := activate_agent.tool_functions.get(
                    function.name)) is not None:
                args = json.loads(function.arguments)

                # 自动注入上下文变量
                if (var_name := activate_agent.use_context_variables.get(
                        function.name)) is not None:
                    args[var_name] = context_variables

                tool_content = tool_function(**args)
                match tool_content:
                    case Agent() as new_agent:
                        result = Result(agent=new_agent,
                                        content=json.dumps(
                                            {"assistant": new_agent.name}))
                    case str() as content:
                        result = Result(content=content)
                    case ContextVariables() as new_variables:
                        result = Result(context_variables=new_variables)
                    case Result() as result:  # 上述三种返回值的组合类
                        pass
                    case _:
                        result = Result()

                activate_agent.add_messages_tool_call(result.content, item.id)
                if result.agent:
                    if activate_agent.llm is not result.agent.llm:  # 底层不是共用模型的话就要拷贝历史消息
                        result.agent.llm.messages = copy.deepcopy(
                            activate_agent.messages)
                    # 切换的 Agent 的初始化操作
                    activate_agent = result.agent
                    activate_agent.llm.instruction = activate_agent.instruction
                # 更新上下文变量
                context_variables.vars.update(result.context_variables.vars)

        assistant_output = activate_agent.llm.chat_with_messages(
            content_only=False,
            tools=activate_agent.tools,
            name=activate_agent.name)

    return Response(agent=activate_agent,
                    content=assistant_output.content,
                    context_variables=context_variables)
