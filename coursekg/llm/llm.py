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

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        body = {
            'model': self.api_type,
            "input": {
                "messages": [{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": message
                }]
            },
            "parameters": {
                "result_format": "message",
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_tokens": self.config.max_tokens,
                "repetition_penalty": self.config.repetition_penalty,
                "presence_penalty": self.config.presence_penalty
            }
        }
        response = requests.post(self.url, headers=headers, json=body)
        return response.json()['output']['choices'][0]['message']['content']


class VLLM(LLM):

    def __init__(
            self,
            path: str,
            stop_token_ids: list[int] = None,
            config: LLMConfig = LLMConfig()) -> None:
        """ 使用VLLM加载模型

        Args:
            path (str): 模型名称或路径
            stop_token_ids (list[int], optional): 停止词表. Defaults to None.
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
        self.stop_token_ids = stop_token_ids

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
            stop_token_ids=self.stop_token_ids)
        messages = [{"role": "user", "content": message}]
        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)

        outputs = self.llm.generate([text], sampling_params)
        return outputs[0].outputs[0].text
