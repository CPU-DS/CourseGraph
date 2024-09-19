# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/llm.py
# Description: 定义大模型类

import os
import requests
from abc import ABC, abstractmethod
import vllm
from vllm import SamplingParams
from .config import LLMConfig
from modelscope import AutoTokenizer
import ollama


class LLM(ABC):

    def __init__(self) -> None:
        """ 多种大模型封装类
        """
        pass

    @abstractmethod
    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入
        
        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            str: 模型输出
        """
        raise NotImplementedError


class QwenAPI(LLM):

    def __init__(
        self,
        api_type: str = 'qwen-max',
        api_key: str = os.getenv("DASHSCOPE_API_KEY"),
        url:
        str = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
        config: LLMConfig = LLMConfig()
    ) -> None:
        """ Qwen 系列模型 API 服务

        Args:
            api_type (str, optional): 模型类型. Defaults to 'qwen-max'.
            api_key (str, optional): API_KEY, 不输入则尝试从环境变量 DASHSCOPE_API_KEY 中获取.
            url (str, optional): 请求地址, 不输入则使用阿里云官方地址.
            config (LLMConfig, optional): 配置. Defaults to LLMConfig().
        """
        super().__init__()
        self.api_type = api_type
        self.api_key = api_key
        self.url = url
        self.config = config
        self.stop = None
        self.tools = None
        self.json: bool = False
        self.messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]

    def _chat_with_messages(self, messages: list) -> dict:
        """ 基于message中保存的历史记录进行对话

        Args:
            messages (list): 历史记录

        Returns:
            dict: json格式的模型返回结果
        """

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        body = {
            'model': self.api_type,
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "message",
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_tokens": self.config.max_tokens,
                "response_format": {
                    "type": "json_object"
                } if self.json else {
                    "type": "text"
                },
                "repetition_penalty": self.config.repetition_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stop": self.stop,
                "tools": self.tools
            }
        }
        response = requests.post(self.url, headers=headers, json=body)
        return response.json()

    def chat(self, message: str, content_only: bool = True) -> str | dict:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入
            content_only (bool, optional): 只返回模型的文本输出. Defaults to True.

        Returns:
            str | dict: 模型输出
        """
        response = self._chat_with_messages(  # 不使用历史记录
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": message
            }])
        content = response['output']['choices'][0]['message']['content']
        return content if content_only else response

    def chat_with_history(self,
                          message: str | None = None,
                          content_only: bool = True) -> str | dict:
        """ 模型的多轮对话

        Args:
            message (str): 用户输入
            content_only (bool, optional): 只返回模型的文本输出. Defaults to True.
        Returns:
            str | dict: 模型输出
        """
        if message is not None:
            self.messages.append({'role': 'user', 'content': message})
        response = self._chat_with_messages(self.messages)
        content = response['output']['choices'][0]['message']['content']
        self.messages.append({'role': 'assistant', 'content': content})
        return content if content_only else response

    def add_message_tool_call(self, tool_content: str, tool_name: str) -> None:
        """ 向历史记录中添加工具调用的历史记录

        Args:
            tool_content (str): 工具调用结果
            tool_name (str): 工具名称
        """
        self.messages.append({
            "role": "tool",
            "content": tool_content,
            "name": tool_content
        })


class VLLM(LLM):

    def __init__(self, path: str, config: LLMConfig = LLMConfig()) -> None:
        """ 使用VLLM加载模型

        Args:
            path (str): 模型名称或路径
            config (LLMConfig, optional): 配置. Defaults to LLMConfig().
        """
        super().__init__()
        self.path = path
        self.config = config
        self.llm = vllm.LLM(
            model=path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        self.stop_token_ids = None
        self.stop_words = None

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            max_tokens=self.config.max_tokens,
            presence_penalty=self.config.presence_penalty,
            stop_token_ids=self.stop_token_ids,
            stop=self.stop_words,
            frequency_penalty=self.config.frequency_penalty)
        messages = [{"role": "user", "content": message}]
        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)

        outputs = self.llm.generate([text], sampling_params)
        return outputs[0].outputs[0].text


class Ollama(LLM):

    def __init__(self, name: str, config: LLMConfig = LLMConfig()) -> None:
        """ 使用ollama加载模型

        Args:
            name (str): 模型名称
            config (LLMConfig, optional): 配置. Defaults to LLMConfig().
        """
        super().__init__()
        self.config = config
        self.name = name
        available_models = [m['name'] for m in ollama.list()['models']]
        if name not in available_models and name:
            ollama.pull(name)
        self.stop = None
        self.json: bool = False

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        return ollama.generate(self.name,
                               prompt=message,
                               format='json' if self.json else '',
                               options={
                                   "temperature": self.config.temperature,
                                   "top_k": self.config.top_k,
                                   "top_p": self.config.top_p,
                                   "repeat_penalty":
                                   self.config.repetition_penalty,
                                   "frequency_penalty":
                                   self.config.frequency_penalty,
                                   "presence_penalty":
                                   self.config.presence_penalty,
                                   "stop": self.stop,
                               })['response']
