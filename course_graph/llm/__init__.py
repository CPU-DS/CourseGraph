# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/__init__.py
# Description: 大模型接口

from .prompt import ExtractPrompt, ExamplePrompt
from .ontology import ontology
from .llm import LLM, VLLM, Qwen, Ollama, OpenAI
from .vl_prompt import vl_prompt
from .prompt_strategy import ExamplePromptStrategy, SentenceEmbeddingStrategy
from .config import llm_config, vlm_config
from .type import Database
from .vlm import VLM
from .parser_prompt import parser_prompt
