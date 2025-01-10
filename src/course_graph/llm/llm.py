# -*- coding: utf-8 -*-
# Create Date: 2024/09/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/llm.py
# Description: 定义兼容 openAI API 的大模型类

import openai
from openai.types.chat import *
from openai import NOT_GIVEN, NotGiven
from abc import ABC, abstractmethod
import os
import requests
import subprocess
import time
import ollama
import base64
import signal
import pathlib
import weakref
from .types import LLMConfig, VLLMConfig
import shlex


class LLM(ABC):

    def __init__(
        self
    ) -> None:
        """ 大模型抽象类
        """
        ABC.__init__(self)

        self.model: str | None = None  # 需要在子类中额外初始化
        self.client: openai.OpenAI | None = None

        self.json: bool = False
        self.stop = None
        self._instruction = 'You are a helpful assistant.'
        
        self.extra_body = {}
        self.config: LLMConfig = {}
        
    @property
    def instruction(self) -> str:
        return self._instruction
    
    @instruction.setter
    def instruction(self, value: str) -> None:
        if len(value) == 0:
            return
        self._instruction = value


    def chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN
    ) -> ChatCompletionMessage:
        """ 基于message中保存的历史消息进行对话, 请在外部保存历史记录, LLM 对象不负责保存

        Args:
            messages (list[ChatCompletionMessageParam]): 历史消息
            tools (list[ChatCompletionToolParam] | NotGiven, optional): 外部tools. Defaults to NOT_GIVEN.
            tool_choice: (ChatCompletionToolChoiceOptionParam | NotGiven, optional): 强制使用外部工具. Defaults to NOT_GIVEN.
            parallel_tool_calls: (bool | NotGiven, optional): 允许工具并行调用. Defaults to NOT_GIVEN.

        Returns:
            ChatCompletionMessage: 模型返回结果
        """
        # functions 废弃
        # 参考: https://platform.openai.com/docs/api-reference/chat/create
        messages = [{'role': 'system', 'content': self.instruction}] + messages
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            top_p=self.config.get('top_p', NOT_GIVEN),
            temperature=self.config.get('temperature', NOT_GIVEN),
            presence_penalty=self.config.get('presence_penalty', NOT_GIVEN),
            frequency_penalty=self.config.get('frequency_penalty', NOT_GIVEN),
            max_tokens=self.config.get('max_tokens', NOT_GIVEN),
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            response_format={
                'type': 'json_object'
            } if self.json else {
                'type': 'text'
            },
            stop=self.stop,
            extra_body={
                'top_k': self.config.get('top_k', NOT_GIVEN),
                'repetition_penalty': self.config.get('repetition_penalty', NOT_GIVEN),
                **self.extra_body
            }).choices[0].message

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str | ChatCompletionMessage: 模型输出
        """
        response = self.chat_completion(messages=[{'role': 'user', 'content': message}])
        return response.content
    
    def image_chat(self, path: str | list[str], message: str) -> str:
        """ 基于图片单轮对话

        Args:
            url (str): 图片路径
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        if isinstance(path, str):
            path = [path]
        content = [
            {
                'type': 'image_url',
                'image_url': {
                    'url': (
                        f"data:image/{pathlib.Path(p).suffix[1:]};base64,{base64.b64encode(open(p, 'rb').read()).decode('utf-8')}"
                        if os.path.exists(p) else p
                    )
                }
            }
            for p in path
        ]

        messages = [{
            'role': "user",
            'content': [
                {'type': 'text', 'text': message},
                *content
            ]
        }]
        response = self.chat_completion(messages=messages)
        return response.content


class OpenAI(LLM):

    def __init__(self,
                 name: str,
                 *,
                 base_url: str = None,
                 api_key: str = None):
        """ OpenAI 模型 API 服务

        Args:
            name (str): 模型名称
            base_url (str, optional): 地址. Defaults to None.
            api_key (str, optional): API key. Defaults to None.
        """
        super().__init__()

        self.model = name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )


class Qwen(OpenAI):

    def __init__(self,
                 name: str = 'qwen-max',
                 *,
                 api_key: str = os.getenv("DASHSCOPE_API_KEY")):
        """ Qwen 系列模型 API 服务

        Args:
            name (str, optional): 模型名称. Defaults to qwen-max.
            api_key (str, optional): API key. Defaults to os.getenv("DASHSCOPE_API_KEY").
        """
        super().__init__(
            name=name,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=api_key)


class DeepSeek(OpenAI):

    def __init__(self,
                 api_key: str = os.getenv("DEEPSEEK_API_KEY")):
        """ DeepSeek 模型 API 服务

        Args:
            api_key (str, optional): API key. Defaults to os.getenv("DEEPSEEK_API_KEY").
        """
        super().__init__(
            name='deepseek-chat',
            base_url='https://api.deepseek.com/v1', 
            api_key=api_key)


class Server:

    def __init__(self,
                 command_list: list[str],
                 test_url: str,
                 log: bool = True,
                 timeout: int = 30):
        """ 启动服务

        Args:
            command_list (list[str]): 命令列表
            test_url (str): 测试地址
            log (bool, optional): 输出控制台日志. Defaults to True.
            timeout (int, optional): 超时时间. Defaults to 30.

        Raises:
            TimeoutError: 服务启动超时
        """
        self.__finalizer = weakref.finalize(self, self.close)
        
        self.process = subprocess.Popen(
            command_list,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ) if not log else subprocess.Popen(command_list) 

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
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            self.process.wait()

           
class Input(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        raise NotImplementedError

    
class Path(Input):
    def __init__(self, path: str):
        super().__init__()
        if self.validate():
            self.path = path
        else:
            raise FileNotFoundError
            
    def validate(self) -> bool:
        return os.path.exists(self.path)

   
class Command(Input):
    def __init__(self, command: str):
        super().__init__()
        if self.validate():
            self.command = command
        else:
            raise ValueError
            
    def validate(self) -> bool:
        return bool(self.command.strip())


class VLLM(LLM, Server):

    def __init__(self,
                 input: Input,
                 *,
                 host: str = 'localhost',
                 port: int = 9017,
                 log: bool = True,
                 timeout: int = 60,
                 config: VLLMConfig = None):
        """ 使用VLLM加载模型

        Args:
            input (Input): 模型路径 Path 或自定义启动命令 Command, 若使用命令启动, 可省略 host, port及 config 中的配置
            host (str, optional): 服务地址. Defaults to 'localhost'.
            port (int, optional): 服务端口. Defaults to 9017.
            timeout (int, optional): 启动服务超时时间. Defaults to 60.
            log (bool, optional): 输出控制台日志. Defaults to True.
            config (VLLMConfig, optional): VLLM配置. Defaults to None.
        """
        LLM.__init__(self)
        
        self.host = host
        self.port = port
        
        match input:
            case Path(path):
                self.model = path
                
                commands = [
                    "vllm", "serve", self.model,
                    "--host", self.host,
                    "--port", str(self.port),
                    "--enable-auto-tool-choice",
                    "--tool-call-parser", "hermes",
                    "--disable-log-requests",
                    "--trust-remote-code"
                ]  
            case Command(command):
                commands = shlex.split(command)
                self.model = commands[2]
                
                if "--host" not in commands:
                    commands.extend([
                        "--host",
                        host
                    ])
                    self.host = host
                else:
                    self.host = commands[commands.index('--host') + 1]
                if "--port" not in commands:
                    commands.extend([
                        "--port",
                        str(port)
                    ])
                    self.port = port
                else:
                    self.port = int(commands[commands.index('--port') + 1])
            case _:
                raise ValueError
        
        for key in config.keys():
            commands.extend([
                f"--{key.replace('_', '-')}",
                str(config[key])
            ])  
    
        Server.__init__(self,
                       command_list=commands,
                       timeout=timeout,
                       log=log,
                       test_url=f'http://{self.host}:{self.port}/health')

        self.client = openai.OpenAI(
            api_key='EMPTY', base_url=f'http://{self.host}:{self.port}/v1')

    def __enter__(self) -> 'VLLM':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class Ollama(LLM, Server):

    def __init__(self,
                 name: str,
                 *,
                 host: str = 'localhost',
                 port: int = 9017,
                 timeout: int = 60):
        """ ollama模型服务

        Args:
            name (str): 模型名称
            timeout (int, optional): 启动服务超时时间. Defaults to 60.
            host (str, optional): 服务地址. Defaults to 'localhost'.
            port (int, optional): 服务端口. Defaults to 9017.
        """
        LLM.__init__(self)
        self.model = name

        available_models = [m['name'] for m in ollama.list()['models']]
        if name not in available_models and name:
            ollama.pull(name)

        self.host = host
        self.port = port

        Server.__init__(self,
                       command_list=['ollama', 'serve'],
                       timeout=timeout,
                       test_url=f'http://{self.host}:{self.port}')
        self.client = openai.OpenAI(
            api_key='EMPTY',
            base_url=f'http://{self.host}:{self.port}/v1',
        )