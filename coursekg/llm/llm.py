# -*- coding: utf-8 -*-
# Create Date: 2024/09/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/llm.py
# Description: 定义兼容 openAI API 的大模型类

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai import NOT_GIVEN
from abc import ABC
from .config import LLMConfig
from typing import Callable
import os
import requests
import subprocess
import time
import shlex
import ollama
from ..agent import Tool


class LLM(ABC):

    def __init__(self, config: LLMConfig = LLMConfig()) -> None:
        """ 大模型抽象类

        Args:
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        ABC.__init__(self)
        self.config = config

        self.model: str | None = None  # 需要在子类中额外初始化
        self.client: OpenAI | None = None

        self.tools: list = []
        self.tool_functions: dict[str, Callable] = {}
        self.json: bool = False
        self.stop = None
        self.messages: list[ChatCompletionMessageParam] = [{
            "role":
            "system",
            "content":
            "You are a helpful assistant."
        }]

    def reset_messages(self) -> None:
        """ 清空历史记录
        """
        self.messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]  # 缺少 name

    def _chat(self,
              messages: list[dict],
              tool: bool = False) -> ChatCompletionMessage:
        """ 基于message中保存的历史记录进行对话

        Args:
            messages (list): 历史记录
            tool (bool, optional): 是否使用tools. Defaults to False.

        Returns:
            ChatCompletionMessage: 模型返回结果
        """
        # functions 废弃
        # 参考: https://platform.openai.com/docs/api-reference/chat/create
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            max_tokens=self.config.max_tokens,
            tools=self.tools if tool and len(self.tools) != 0 else NOT_GIVEN,
            response_format={
                'type': 'json_object'
            } if self.json else {
                'type': 'text'
            },
            stop=self.stop,
            extra_body={
                'top_k': self.config.top_k
            }).choices[0].message

    def chat(self,
             message: str,
             content_only: bool = True) -> str | ChatCompletionMessage:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入
            content_only (bool, optional): 只返回模型的文本输出. Defaults to True.

        Returns:
            str | ChatCompletionMessage: 模型输出
        """
        response = self._chat(
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": message
            }])
        return response.content if content_only else response

    def chat_with_messages(self,
                           message: str | None = None,
                           content_only: bool = True,
                           tool: bool = False) -> str | ChatCompletionMessage:
        """ 模型的多轮对话

        Args:
            message (str): 用户输入
            content_only (bool, optional): 只返回模型的文本输出. Defaults to True.
            tool (bool, optional): 是否使用tools. Defaults to False.
        Returns:
            str | ChatCompletionMessage: 模型输出
        """
        if message is not None:
            self.messages.append({
                'role': 'user',
                'content': message
            })  # 缺少 name
        response = self._chat(self.messages, tool=tool)
        self.messages.append(response.model_dump(
        ))  # ChatCompletionMessage -> ChatCompletionMessageParam 缺少 name
        return response.content if content_only else response

    def _add_message_tool_call(self, tool_content: str,
                               tool_call_id: str) -> None:
        """ 向历史记录中添加工具调用的历史记录

        Args:
            tool_content (str): 工具调用结果
            tool_call_id (str): 工具id
        """
        self.messages.append({
            "role": "tool",
            "content": tool_content,
            "tool_call_id": tool_call_id
        })

    def add_tools(self, *tools: Tool) -> 'LLM':
        """ 添加外部工具函数
        """
        for tool in tools:
            self.tools.append(tool["tool"])
            self.tool_functions.update({
                tool.get('function_name', tool["function"].__name__):
                tool["function"]
            })
        return self

    def chat_with_tools(self, message: str) -> str:
        """ 调用外部函数进行对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        assistant_output = self.chat_with_messages(message,
                                                   content_only=False,
                                                   tool=True)
        while assistant_output.tool_calls:  # None 或者空数组
            functions = assistant_output.tool_calls
            for item in functions:
                function = item.function
                tool_function = self.tool_functions.get(function.name)
                if tool_function:
                    tool_content = tool_function(**eval(function.arguments))
                    self._add_message_tool_call(tool_content, item.id)
            assistant_output = self.chat_with_messages(content_only=False,
                                                       tool=True)
        return assistant_output.content


class Qwen(LLM):

    def __init__(self,
                 name: str = 'qwen-max',
                 config: LLMConfig = LLMConfig()):
        """ Qwen 系列模型 API 服务

        Args:
            name (str): 模型名称
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        super().__init__(config)

        self.model = name
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


class Serve:

    def __init__(self,
                 command_list: list[str],
                 test_url: str,
                 timeout: int = 30):
        """ 启动服务

        Args:
            command_list (list[str]): 命令列表
            test_url (str): 测试地址
            timeout (int, optional): 超时时间. Defaults to 30.

        Raises:
            TimeoutError: 服务启动超时
        """
        self.process = subprocess.Popen(command_list)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(test_url)
                if response.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError

    def close(self):
        """ 关闭服务
        """
        if self.process:
            self.process.terminate()
            self.process.wait()


class VLLM(LLM, Serve):

    def __init__(self,
                 path: str,
                 config: LLMConfig = LLMConfig(),
                 starting_command: str = None):
        """使用VLLM加载模型

        Args: path (str): 模型名称或路径 config (LLMConfig, optional): 配置. Defaults to LLMConfig(). starting_command (str,
        optional): VLLM启动命令 (适合于需要自定义template的情况), 也可以使用默认命令, LLMConfig中的配置会自动加入. Defaults to None.
        """
        LLM.__init__(self, config)

        self.model = path
        if starting_command is None:
            command_list = shlex.split(f"""vllm serve {self.model}\
                                            --host localhost\
                                            --port 9017\
                                            --gpu-memory-utilization {str(self.config.gpu_memory_utilization)}\
                                            --tensor-parallel-size {str(self.config.tensor_parallel_size)}\
                                            --max-model-len {str(self.config.max_model_len)}\
                                            --enable-auto-tool-choice\
                                            --tool-call-parser hermes\
                                            --disable-log-requests""")
            self.host = 'localhost'
            self.port = '9017'
        else:
            command_list = shlex.split(starting_command)
            if "--gpu-memory-utilization" not in command_list:
                command_list.extend([
                    "--gpu-memory-utilization",
                    str(self.config.gpu_memory_utilization)
                ])
            if "--tensor-parallel-size" not in command_list:
                command_list.extend([
                    "--tensor-parallel-size",
                    str(self.config.tensor_parallel_size)
                ])
            if "--max-model-len" not in command_list:
                command_list.extend(
                    ["--max-model-len",
                     str(self.config.max_model_len)])
            try:
                idx = command_list.index('--host')
                self.host = command_list[idx + 1]
            except ValueError:
                self.host = '0.0.0.0'
            try:
                idx = command_list.index('--port')
                self.port = command_list[idx + 1]
            except ValueError:
                self.port = '8000'

        Serve.__init__(self,
                       command_list=command_list,
                       timeout=60,
                       test_url=f'http://{self.host}:{self.port}/health')

        self.client = OpenAI(api_key='EMPTY',
                             base_url=f'http://{self.host}:{self.port}/v1')


class Ollama(LLM, Serve):

    def __init__(self, name: str, config: LLMConfig = LLMConfig()):
        """ ollama模型服务

        Args:
            name (str): 模型名称
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        LLM.__init__(self, config)
        self.model = name

        available_models = [m['name'] for m in ollama.list()['models']]
        if name not in available_models and name:
            ollama.pull(name)

        self.host = 'localhost'
        self.port = 11434

        Serve.__init__(self,
                       command_list=['ollama', 'serve'],
                       timeout=60,
                       test_url=f'http://{self.host}:{self.port}')
        self.client = OpenAI(
            api_key='EMPTY',
            base_url=f'http://{self.host}:{self.port}/v1',
        )
