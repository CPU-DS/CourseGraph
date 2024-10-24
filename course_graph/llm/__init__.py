# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/__init__.py
# Description: 大模型接口

from .prompt import ExtractPrompt, ExamplePrompt, ParserPrompt
from .llm import LLM, VLLM, Qwen, Ollama, OpenAI
from .mlm import MLM
from .vision_prompt import MiniCPMPrompt, VLUPrompt
from .prompt_strategy import ExamplePromptStrategy, SentenceEmbeddingStrategy
from .config import VisualConfig, LLMConfig
from .type import Database
