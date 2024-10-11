# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/__init__.py
# Description: 大模型接口

from .prompt import IEPrompt, ExamplePrompt, ParserPrompt
from .llm import LLM, VLLM, Qwen, Ollama
from .mlm import MiniCPM, MLM
from .vision_prompt import MiniCPMPrompt, Interaction, VLUPrompt
from .prompt_strategy import ExamplePromptStrategy, SentenceEmbeddingStrategy
from .config import VisualConfig, LLMConfig
from .type import Model, Database
