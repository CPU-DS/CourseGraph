# -*- coding: utf-8 -*-
# Create Date: 2024/09/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/llm.py
# Description: 定义兼容 OpenAI API 的大模型类

from openai import OpenAI
from openai.types.chat import *
from openai.types import *
from openai import NOT_GIVEN, NotGiven
from typing import Generator
import os
import requests
import subprocess
import time
import base64
import signal
from pathlib import Path
import weakref
from .config import LLMConfig, VLLMConfig
import shlex
from contextlib import contextmanager
import re


class LLMBase:

    def __init__(
        self,
        base_url: str,
        api_key: str,
        proxy: str = None
    ):
        """ 大模型基类, 兼容 OpenAI API
            
        Args:
            base_url (str): OpenAI Base URL
            api_key (str): OpenAI API Key
            proxy (str, optional): 代理. Defaults to None.
        """
        self.proxy = proxy
        
        with self._proxy_context():
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        
        self.model = ''

        self.extra_body = {}
        self.config: LLMConfig = {}
        
        self._instruction = 'You are a helpful assistant.'
        
    @property
    def instruction(self) -> str:
        return self._instruction
    
    @instruction.setter
    def instruction(self, value: str) -> None:
        if len(value) == 0:
            return
        self._instruction = value

    @contextmanager
    def _proxy_context(self):
        if self.proxy:
            os.environ['http_proxy'] = self.proxy
            os.environ['https_proxy'] = self.proxy
        try:
            yield
        finally:
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)

    def chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        stream: bool = False
    ) -> ChatCompletion:
        """ 基于message中保存的历史消息进行对话, 请在外部保存历史记录, LLM 对象不负责保存

        Args:
            messages (list[ChatCompletionMessageParam]): 历史消息
            tools (list[ChatCompletionToolParam] | NotGiven, optional): 外部tools. Defaults to NOT_GIVEN.
            tool_choice: (ChatCompletionToolChoiceOptionParam | NotGiven, optional): 强制使用外部工具. Defaults to NOT_GIVEN.
            parallel_tool_calls: (bool | NotGiven, optional): 允许工具并行调用. Defaults to NOT_GIVEN.
            stream: (bool, optional): 是否流式输出. Defaults to False.

        Returns:
            ChatCompletion: 模型返回结果
        """
        messages = [{'role': 'system', 'content': self.instruction}] + messages
        with self._proxy_context():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                top_p=self.config.get('top_p', NOT_GIVEN),
                temperature=self.config.get('temperature', NOT_GIVEN),
                presence_penalty=self.config.get('presence_penalty', NOT_GIVEN),
                frequency_penalty=self.config.get('frequency_penalty', NOT_GIVEN),
                max_tokens=self.config.get('max_tokens', NOT_GIVEN),
                tools=tools,
                reasoning_effort=self.config.get('reasoning_effort', NOT_GIVEN),
                parallel_tool_calls=parallel_tool_calls,
                tool_choice=tool_choice,
                response_format={
                    'type': 'json_object'
                } if self.config.get('json', False) else NOT_GIVEN,
                stop=self.config.get('stop', NOT_GIVEN),
                extra_body={
                    'top_k': self.config.get('top_k', NOT_GIVEN),
                    'repetition_penalty': self.config.get('repetition_penalty', NOT_GIVEN),
                    **self.extra_body
                })
        return response

    def embedding(self, input: str, dimensions: int = 1024, encoding_format: str = "float") -> list:
        """ 文本嵌入

        Args:
            input (str): 输入文本
            dimensions (int, optional): 维度. Defaults to 1024.
            encoding_format (str, optional): 编码格式. Defaults to "float".

        Returns:
            list: 向量
        """
        with self._proxy_context():
            response = self.client.embeddings.create(
                model=self.model,
                input=input,
                dimensions=dimensions,
                encoding_format=encoding_format
            ).data[0].embedding
        return response


class LLM(LLMBase):

    def __init__(self,
                 api_key: str,
                 base_url: str = None,
                 proxy: str = None):
        """ 大模型封装类

        Args:
            api_key (str): API key.
            base_url (str, optional): 地址. Defaults to None.
            proxy (str, optional): 代理. Defaults to None.
        """ 
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            proxy=proxy
        )
        
    def get_model_ids(self) -> list[str]:
        """ 获取模型列表

        Returns:
            list[str]: 模型列表
        """
        return [model.id for model in self.client.models.list().data]
    
    def get_model_config(self) -> dict:
        """ 获取模型配置
        
        Returns:
            dict: 配置信息
        """
        return {**self.config, **self.extra_body}
    
    def chat(self, message: str) -> tuple[str, str] | tuple[str, None]:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            tuple[str, str] | tuple[str, None]: 模型输出, 推理过程
        """
        response = self.chat_completion(messages=[{'role': 'user', 'content': message}]).choices[0].message
        resp = response.content
        reasoning = response.reasoning_content if hasattr(response, 'reasoning_content') else None
        if reasoning is None and self.config.get('reasoning_parser') is not None:
            match self.config.get('reasoning_parser'):
                case 'deepseek_r1':
                    pattern = r'<think>(.*?)</think>'
                    reasoning = re.findall(pattern, resp, re.DOTALL)[0]
                    resp = re.sub(pattern, '', resp, flags=re.DOTALL)
                case _:
                    ...
        return resp.strip(), reasoning.strip() if reasoning is not None else None
    
    def stream_chat(self, message: str) -> Generator[str, None, None]:
        """ 模型的单轮流式对话
        
        Args:
            message (str): 用户输入

        Returns:
            Generator[str, None, None]: 模型输出
        """
        chunks = self.chat_completion(messages=[{'role': 'user', 'content': message}], stream=True)
        for chunk in chunks:
            choices = chunk.choices
            if len(choices) > 0:
                response = chunk.choices[0].delta
                content = response.reasoning_content if hasattr(response, 'reasoning_content') and response.reasoning_content is not None else response.content
                if content:
                    yield content
    
    def image_chat(self, path: str | list[str], message: str) -> tuple[str, str] | tuple[str, None]:
        """ 基于图片单轮对话

        Args:
            url (str): 图片路径
            message (str): 用户输入

        Returns:
            tuple[str, str] | tuple[str, None]: 模型输出, 推理过程
        """
        if isinstance(path, str):
            path = [path]
        content = [
            {
                'type': 'image_url',
                'image_url': {
                    'url': (
                        f"data:image/{Path(p).suffix[1:]};base64,{base64.b64encode(open(p, 'rb').read()).decode('utf-8')}"
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
        response = self.chat_completion(messages=messages).choices[0].message
        return response.content, None


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


class VLLM(Server):

    def __init__(self,
                 path: str,
                 *,
                 host: str = 'localhost',
                 port: int = 9017,
                 log: bool = True,
                 timeout: int = 60,
                 config: VLLMConfig = None):
        """ 使用VLLM加载模型

        Args:
            path (str): 模型路径
            host (str, optional): 服务地址. Defaults to 'localhost'.
            port (int, optional): 服务端口. Defaults to 9017.
            timeout (int, optional): 启动服务超时时间. Defaults to 60.
            log (bool, optional): 输出控制台日志. Defaults to True.
            config (VLLMConfig, optional): VLLM配置. Defaults to None.
        """
        
        self.host = host
        self.port = port
        
        self.model = path
        
        command = f"""
            vllm serve {self.model}\
                --host {self.host}\
                --port {self.port}\
                --disable-log-requests\
                --trust-remote-code
        """
        commands = shlex.split(command)
            
        if config:
            for key in config.keys():
                if type(config[key]) == bool and config[key]:
                    commands.extend([
                        f"--{key.replace('_', '-')}"
                    ])
                else:
                    commands.extend([
                        f"--{key.replace('_', '-')}",
                        str(config[key])
                    ])

        Server.__init__(self,
                       command_list=commands,
                       timeout=timeout,
                       log=log,
                       test_url=f'http://{self.host}:{self.port}/health')

    def to_llm(self) -> LLM:
        """ 获取一个 LLM 对象

        Returns:
            LLM: LLM 对象
        """
        return LLM(
            api_key='EMPTY',
            base_url=f'http://{self.host}:{self.port}/v1'
        )

    def run_loop(self):
        """ 持续运行直到终止
        """
        def signal_handler(signum, frame):
            self.close()          
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            signal.pause()
        except KeyboardInterrupt:
            self.close()

    def __enter__(self) -> 'VLLM':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

