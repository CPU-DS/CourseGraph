# -*- coding: utf-8 -*-
# Create Date: 2024/10/14
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/agent.py
# Description: 定义智能体

from ..llm import LLMBase
from .types import ContextVariables, Tool
from openai.types.chat import *
import inspect
import docstring_parser
from typing import Callable, Awaitable, Literal
from os import PathLike
from openai import NOT_GIVEN
from .mcp import MCPServer
from shortuuid import uuid


class Agent:

    def __init__(
            self,
            llm: LLMBase,
            name: str = 'Assistant',
            functions: list[Callable | Awaitable] = None,
            tool_choice: str | Literal['required', 'auto', 'none'] = 'auto',
            parallel_tool_calls: bool = False,
            instruction: str | Callable[..., str] | PathLike = 'You are a helpful assistant.',
            instruction_args: dict = None,
            mcp_server: list[MCPServer] = None,
            mcp_impl: Literal['function_call'] = 'function_call'
    ) -> None:
        """ 智能体类

        Args: 
            llm (LLMBase): 大模型 
            name (str, optional): 名称. Defaults to 'Assistant'.
            functions: (list[Callable | Awaitable], optional): 工具函数. Defaults to None.
            parallel_tool_calls: (bool, optional): 允许工具并行调用. Defaults to False.
            tool_choice: (Literal['required', 'auto', 'none'], optional). 强制使用工具函数, 选择模式或提供函数名称. Defaults to 'auto'.
            instruction (str | Callable[Any, str] | PathLike, optional): 指令. Defaults to 'You are a helpful assistant.'.
            instruction_args: (dict, optional): 指令参数. Defaults to {}.
            mcp_server: (list[MCPServer], optional): MCP 服务器. Defaults to None.
            mcp_impl: (Literal['function_call'] | NotGiven, optional): MCP 协议实现方式, 目前只支持 'function_call'. Defaults to 'function_call'.
        """
        self.id = str(uuid())
        self.llm = llm
        self.name = name
        self.instruction = instruction
        self.instruction_args = instruction_args
        self.tools: list[ChatCompletionToolParam] = []  # for LLM
        
        self.tool_functions: dict[str, Callable | Awaitable] = {}  # for local function call
        self.mcp_functions: dict[str, MCPServer] = {}  # for remote function call
        
        self.parallel_tool_calls = parallel_tool_calls
        self.use_context_variables: dict[str, str] = {}  # 使用了上下文变量的函数以及相应的形参名称
        self.use_agent_variables: dict[str, str] = {}  # 使用了Agent变量的函数以及相应的形参名称

        if functions:
            self.add_tool_functions(*functions)
        

        if tool_choice not in ['required', 'auto', 'none']:  # 需要注意工具无限循环
            self.tool_choice = {
                "type": "function",
                "function": {
                    "name": tool_choice
                }
            }
        else:
            self.tool_choice = tool_choice

        self.messages: list[ChatCompletionMessageParam] = []
        if mcp_server:
            for server in mcp_server:
                tools = server.tools
                for tool in tools:
                    self.tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': tool.description,
                            'parameters': tool.inputSchema
                        }
                    })  # 注意不能使用 add_tools 方法
                    self.mcp_functions[tool.name] = server

    def chat_completion(self, message: str = None) -> ChatCompletionMessage:
        """ Agent 多轮对话

        Args:
            message (str): 用户输入

        Returns:
            ChatCompletionMessage: 模型输出
        """
        
        if message is not None:
            self.add_user_message(message)

        # 协调参数类型关系
        if len(self.tools) == 0:
            tools = NOT_GIVEN
        else:
            tools = self.tools

        response = self.llm.chat_completion(
            self.messages,
            parallel_tool_calls=self.parallel_tool_calls,
            tools=tools,
            stream=False,
            tool_choice=self.tool_choice).choices[0].message
        # 保存历史记录
        resp = response.model_dump()
        resp['name'] = self.name
        self.messages.append(resp)  # 比 add_assistant_message 信息更详细
        return response

    def add_user_message(self, message: str) -> None:
        """ 添加用户记录

        Args:
            message (str): 用户输入
        """
        message = {'role': 'user', 'content': message}
        self.messages.append(message)

    def add_assistant_message(self, message: str, name: str = None) -> None:
        """ 添加模型记录

        Args:
            message (str): 模型输出
            name (str): 模型名称. Defaults to self.name.
        """
        message = {'content': message, 'role': 'assistant', 'name': name or self.name}
        self.messages.append(message)

    def add_tool_call_message(self, tool_content: str, tool_call_id: str) -> None:
        """ 添加工具调用记录

        Args:
            tool_content (str): 工具调用结果
            tool_call_id (str): 工具id

        """
        message = {
            "role": "tool",
            "content": tool_content,
            "tool_call_id": tool_call_id
        }
        self.messages.append(message)
    
    def tool(self) -> Callable | Awaitable:
        """ 标记一个外部工具函数
        """
        def wrapper(function: Callable | Awaitable) -> Callable | Awaitable:
            self.add_tool_functions(function)
            return function
        return wrapper
    
    def add_tools(self, *tools: 'Tool') -> 'Agent':
        """ 添加外部工具

        Args:
            *tools (Tool): 外部工具

        Returns:
            Agent: 智能体
        """
        for tool in tools:
            self.tools.append(tool["tool"])
            function = tool["function"]
            function_name = tool.get('function_name', function.__name__)
            if function_name == '<lambda>':
                continue
            self.tool_functions[function_name] = function
            if (r := tool.get('context_variables_parameter_name')) is not None:
                self.use_context_variables[function_name] = r
            if (r := tool.get('context_agent_parameter_name')) is not None:
                self.use_agent_variables[function_name] = r
        return self

    def add_tool_functions(self, *functions: Callable | Awaitable) -> 'Agent':
        """ 添加外部工具函数，并从自动解析函数描述、参数类型、以及参数描述等信息 \n
        若使用了不支持的文档风格或使用复杂参数，请调用 add_tools 方法手动编写函数描述

        Args:
            *functions (Callable): 外部工具函数

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
                continue
            docstring = inspect.getdoc(function)
            res = docstring_parser.parse(docstring)

            description = (res.short_description or '') + ('\n' + res.long_description if res.long_description else '')

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
                # 和ContextVariables机制相同
                if p.annotation == Agent:
                    del properties[arg_name]
                    self.use_agent_variables[function_name] = arg_name
                    continue
                # 签名中的参数类型
                if p.annotation != inspect._empty:
                    properties[arg_name]['type'] = type_map.get(
                        str(p.annotation).replace("<class '", '').replace("'>", ''), 'any')
                # 签名中的默认值
                if p.default == inspect._empty:
                    required.append(arg_name)

            for doc_parameters in res.params:
                if doc_parameters.arg_name in properties:
                    # 文档中的参数类型 (优先使用签名中的参数类型)
                    if doc_parameters.type_name is not None and 'type' not in properties[doc_parameters.arg_name]:
                        properties[doc_parameters.arg_name]['type'] = type_map.get(doc_parameters.type_name, 'any')
                    # 文档中的参数描述
                    if doc_parameters.description is not None:
                        properties[doc_parameters.arg_name]['description'] = doc_parameters.description
                    # 文档中的是否可选 (签名中提示required这里也不会丢掉)
                    if not doc_parameters.is_optional and doc_parameters.arg_name not in required:
                        required.append(doc_parameters.arg_name)

            # 如果签名和文档中都未标注类型则可以使用默认值类型替代
            for arg_name, p in signature_parameters.items():
                if arg_name in properties:
                    if p.default != inspect._empty and 'type' not in properties[arg_name]:
                        properties[arg_name]['type'] = type_map.get(
                            str(type(p.default)).replace("<class '",'').replace("'>", ''),'any')

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
    
    def __repr__(self) -> str:
        return f'Agent(id={self.id}, name={self.name})'