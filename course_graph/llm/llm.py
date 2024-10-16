# -*- coding: utf-8 -*-
# Create Date: 2024/09/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coucourse_graphrsekg/llm/llm.py
# Description: 定义兼容 openAI API 的大模型类

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam, ChatCompletionToolParam
from openai import NOT_GIVEN, NotGiven
from abc import ABC
from .config import LLMConfig
import os
import requests
import subprocess
import time
import shlex
import ollama


class LLM(ABC):

    def __init__(
        self,
        instruction: str = 'You are a helpful assistant.',
        config: LLMConfig = LLMConfig()
    ) -> None:
        """ 大模型抽象类

        Args:
            instruction (str, optional): 指令. Defaults to 'You are a helpful assistant.'.
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        ABC.__init__(self)
        self.config = config

        self.model: str | None = None  # 需要在子类中额外初始化
        self.client: OpenAI | None = None

        self.json: bool = False
        self.stop = None
        self.instruction = instruction
        self.messages: list[ChatCompletionMessageParam] = []  # 不包括instruction

    def _chat(
        self, messages: list[dict],
        tools: list[ChatCompletionToolParam] | NotGiven
    ) -> ChatCompletionMessage:
        """ 基于message中保存的历史记录进行对话

        Args:
            messages (list): 历史记录
            tools (list[ChatCompletionToolParam] | NotGiven): 外部tools.

        Returns:
            ChatCompletionMessage: 模型返回结果
        """
        # functions 废弃
        # 参考: https://platform.openai.com/docs/api-reference/chat/create
        messages = [{'role': 'system', 'content': self.instruction}] + messages
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            max_tokens=self.config.max_tokens,
            tools=tools,
            response_format={
                'type': 'json_object'
            } if self.json else {
                'type': 'text'
            },
            stop=self.stop,
            extra_body={
                'top_k': self.config.top_k
            }).choices[0].message

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str | ChatCompletionMessage: 模型输出
        """
        response = self._chat(messages=[{
            'role': 'user',
            'content': message
        }],
                              tools=NOT_GIVEN)
        return response.content

    def chat_with_messages(
        self,
        message: str | None = None,
        name: str | None = None,
        content_only: bool = True,
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN
    ) -> str | ChatCompletionMessage:
        """ 模型的多轮对话

        Args:
            message (str): 用户输入
            name (str): 名称信息
            content_only (bool, optional): 只返回模型的文本输出. Defaults to True.
            tools (list[ChatCompletionToolParam] | NotGiven): 外部tools. Defaults to NOT_GIVEN.
        Returns:
            str | ChatCompletionMessage: 模型输出
        """
        if message is not None:
            self.messages.append({'role': 'user', 'content': message})
        response = self._chat(self.messages, tools=tools)
        resp = response.model_dump()
        if name is not None:
            resp['name'] = name
        self.messages.append(resp)
        return response.content if content_only else response


class API(LLM):

    def __init__(self,
                 name: str,
                 *,
                 base_url: str,
                 api_key: str,
                 config: LLMConfig = LLMConfig()):
        """ OpenAI 模型 API 服务

        Args:
            name (str): 模型名称
            base_url (str): 地址
            api_key (str): API key.
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        super().__init__(config=config)

        self.model = name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )


class Qwen(API):

    def __init__(self,
                 name: str = 'qwen-max',
                 *,
                 api_key: str = os.getenv("DASHSCOPE_API_KEY"),
                 config: LLMConfig = LLMConfig()):
        """ Qwen 系列模型 API 服务

        Args:
            name (str, optional): 模型名称. Defaults to qwen-max.
            api_key (str, optional): API key. Defaults to os.getenv("DASHSCOPE_API_KEY").
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        super().__init__(
            name=name,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=api_key,
            config=config)


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

    def __enter__(self) -> 'Serve':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class VLLM(LLM, Serve):

    def __init__(self,
                 path: str,
                 *,
                 config: LLMConfig = LLMConfig(),
                 starting_command: str = None):
        """ 使用VLLM加载模型

        Args:
            path (str): 模型名称或路径
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
            starting_command (str, optional): VLLM启动命令 (适合于需要自定义template的情况), 也可以使用默认命令, LLMConfig中的配置会自动加入. Defaults to None.
        """
        LLM.__init__(self, config=config)

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

    def __init__(self, name: str, *, config: LLMConfig = LLMConfig()):
        """ ollama模型服务

        Args:
            name (str): 模型名称
            config (LLMConfig, optional): 大模型配置. Defaults to LLMConfig().
        """
        LLM.__init__(self, config=config)
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
