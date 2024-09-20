# -*- coding: utf-8 -*-
# Create Date: 2024/09/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/llm.py
# Description: 定义兼容 openAI API 的大模型类

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai import NOT_GIVEN
from abc import ABC
from .config import LLMConfig
import os
import requests
import subprocess
import time
import shlex


class LLM(ABC):

    def __init__(self, config: LLMConfig = LLMConfig()) -> None:
        """ 大模型抽象类

        Args:
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        super().__init__()
        self.config = config

        self.model: str | None = None
        self.client: OpenAI | None = None

        self.tools: list = []
        self.tool_functions: dict = {}
        self.json: bool = False
        self.stop = None
        self.messages: list[dict] = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]

    def reset_messages(self) -> None:
        """ 清空历史记录
        """
        self.messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]

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
            self.messages.append({'role': 'user', 'content': message})
        response = self._chat(self.messages, tool=tool)
        self.messages.append(response.model_dump())
        return response.content if content_only else response

    def _add_message_tool_call(self, tool_content: str,
                               tool_name: str) -> None:
        """ 向历史记录中添加工具调用的历史记录

        Args:
            tool_content (str): 工具调用结果
            tool_name (str): 工具名称
        """
        self.messages.append({
            "role": "tool",
            "content": tool_content,
            "name": tool_name
        })

    def add_tool_functions(self, *functions, **function_kwargs):
        """ 添加工具函数以供模型调用
        """
        self.tool_functions.update({tool.__name__: tool for tool in functions})
        # 允许通过关键字的方式自定义工具名称
        self.tool_functions.update(function_kwargs)

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
                    tool_name = function.name
                    self._add_message_tool_call(tool_content, tool_name)
            assistant_output = self.chat_with_messages(content_only=False,
                                                       tool=True)
        return assistant_output.content

    def close(self):
        """ 关闭模型, 子类如果需要可以重载
        """
        pass


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


class VLLM(LLM):

    def __init__(self,
                 path: str,
                 config: LLMConfig = LLMConfig(),
                 starting_command: str = None):
        """使用VLLM加载模型

        Args: path (str): 模型名称或路径 config (LLMConfig, optional): 配置. Defaults to LLMConfig(). starting_command (str,
        optional): VLLM启动命令 (适合于需要自定义template的情况), 也可以使用默认命令, LLMConfig中的配置会自动加入. Defaults to None.
        """
        super().__init__(config)

        self.model = path
        if starting_command is None:
            command_list = shlex.split(f"""vllm serve {self.model}\
                                            --host 127.0.0.1\
                                            --port 9017\
                                            --gpu-memory-utilization {str(self.config.gpu_memory_utilization)}\
                                            --tensor-parallel-size {str(self.config.tensor_parallel_size)}\
                                            --max-model-len {str(self.config.max_model_len)}\
                                            --enable-auto-tool-choice\
                                            --tool-call-parser hermes\
                                            --disable-log-requests""")
            self.host = '127.0.0.1'
            self.port = '9017'
        else:
            command_list = shlex.split(starting_command)
            if "--gpu-memory-utilization" not in command_list:
                command_list.extend(["--gpu-memory-utilization",
                                    str(self.config.gpu_memory_utilization)])
            if "--tensor-parallel-size" not in command_list:
                command_list.extend(["--tensor-parallel-size",
                                    str(self.config.tensor_parallel_size)])
            if "--max-model-len" not in command_list:
                command_list.extend(["--max-model-len",
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

        self.vllm_process = subprocess.Popen(command_list)
        self._wait_for_vllm()

        self.client = OpenAI(api_key='EMPTY',
                             base_url=f'http://{self.host}:{self.port}/v1')

    def _wait_for_vllm(self, timeout: int = 30):
        """ 等待VLLM服务启动

        Args:
            timeout (int, optional): 超时时间. Defaults to 30.

        Raises:
            TimeoutError: 超时
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}/health")
                if response.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError

    def close(self):
        """ 关闭VLLM进程
        """
        if self.vllm_process:
            self.vllm_process.terminate()
            self.vllm_process.wait()
