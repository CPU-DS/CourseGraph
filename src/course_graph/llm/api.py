# -*- coding: utf-8 -*-
# Create Date: 2025/02/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/api.py
# Description: 三方大模型 API 接口

from .llm import LLM
import os
from typing import Literal



class Qwen(LLM):
    def __init__(self,
                 api_key: str = os.getenv('DASHSCOPE_API_KEY')):
        """ 阿里云大模型 API 服务

        Args:
            api_key (str, optional): API key. Defaults to os.getenv('DASHSCOPE_API_KEY').
        """
        super().__init__(
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=api_key)
        self.model = 'qwen-max'
        

class DeepSeek(LLM):
    def __init__(self,
                 api_key: str = os.getenv('DEEPSEEK_API_KEY')):
        """ DeepSeek 模型 API 服务

        Args:
            api_key (str, optional): API key. Defaults to os.getenv('DEEPSEEK_API_KEY').
        """
        super().__init__(
            base_url='https://api.deepseek.com/v1', 
            api_key=api_key)
        self.model: Literal['deepseek-chat', 'deepseek-reasoner'] = 'deepseek-chat'


class OpenRouter(LLM):
    def __init__(self,
                 api_key: str = os.getenv('OPENROUTER_API_KEY')):
        """ OpenRouter API 服务

        Args:
            api_key (str, optional): API key. Defaults to os.getenv('OPENROUTER_API_KEY').
        """
        super().__init__(
            base_url='https://openrouter.ai/api/v1', 
            api_key=api_key)


class Volcengine(LLM):
    def __init__(self,
                 api_key: str = os.getenv('ARK_API_KEY')):
        """ 火山引擎 API 服务
        
        Args:
            api_key (str, optional): API key. Defaults to os.getenv('ARK_API_KEY').
        """
        super().__init__(
            base_url='https://ark.cn-beijing.volces.com/api/v3',
            api_key=api_key)
