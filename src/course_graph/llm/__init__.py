# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/__init__.py
# Description: 大模型接口

from .llm import LLM, VLLM, Ollama, OpenAI
from .api import *
from .config import LLMConfig, VLLMConfig
from .ontology import ONTOLOGY
