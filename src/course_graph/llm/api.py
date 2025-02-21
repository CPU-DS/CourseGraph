# -*- coding: utf-8 -*-
# Create Date: 2025/02/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/api.py
# Description: 三方大模型 API 接口

from .llm import OpenAI
import os
from typing import Literal


class Qwen(OpenAI):

    def __init__(self,
                 name: str = 'qwen-max',
                 *,
                 api_key: str = os.getenv('DASHSCOPE_API_KEY')):
        """ 阿里云大模型 API 服务

        Args:
            name (str, optional): 模型名称. Defaults to qwen-max.
            api_key (str, optional): API key. Defaults to os.getenv('DASHSCOPE_API_KEY').
        """
        super().__init__(
            name=name,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=api_key)


class DeepSeek(OpenAI):

    def __init__(self,
                 name: Literal['deepseek-chat', 'deepseek-reasoner'] = 'deepseek-chat',
                 *,
                 api_key: str = os.getenv('DEEPSEEK_API_KEY')):
        """ DeepSeek 模型 API 服务

        Args:
            name (Literal['deepseek-chat', 'deepseek-reasoner'], optional): 模型名称. Defaults to deepseek-chat.
            api_key (str, optional): API key. Defaults to os.getenv('DEEPSEEK_API_KEY').
        """
        super().__init__(
            name=name,
            base_url='https://api.deepseek.com/v1', 
            api_key=api_key)


class OpenRouter(OpenAI):
    def __init__(self, 
                 name, 
                 *, 
                 api_key: str = os.getenv('OPENROUTER_API_KEY')):
        """ OpenRouter API 服务

        Args:
            name (str): 模型名称.
            api_key (str, optional): API key. Defaults to os.getenv('OPENROUTER_API_KEY').
        """
        super().__init__(
            name=name, 
            base_url='https://openrouter.ai/api/v1', 
            api_key=api_key)
