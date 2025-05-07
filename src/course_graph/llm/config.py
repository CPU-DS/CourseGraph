# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/type.py
# Description: 定义大模型配置

from typing import TypedDict, Literal
from pydantic import BaseModel


class LLMConfig(TypedDict, total=False):
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    reasoning_effort: Literal["low", "medium", "high"]
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float
    json: bool | BaseModel
    stop: list[str]


class VLLMConfig(TypedDict, total=False):
    gpu_memory_utilization: float
    tensor_parallel_size: int
    pipeline_parallel_size: int
    max_model_len: int
    enable_auto_tool_choice: bool
    tool_call_parser: str
    enable_reasoning: bool
    reasoning_parser: Literal['deepseek_r1']
    chat_template: str