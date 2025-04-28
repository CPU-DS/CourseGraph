# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/__init__.py
# Description: 提示词相关

from .prompt import ExamplePrompt, Prompt
from .prompt_strategy import ExamplePromptStrategy, SentenceEmbeddingStrategy
from .vl_prompt import VLPrompt
from .parser_prompt import ParserPrompt
from .utils import post_process
