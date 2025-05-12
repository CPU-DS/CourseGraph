# -*- coding: utf-8 -*-
# Create Date: 2025/05/12
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/filter.py
# Description: 提示词示例过滤

from .prompt import Prompt
from ..llm import LLM
from abc import ABC, abstractmethod
from typing import Callable, Any, Literal


class Filter(ABC):

    @abstractmethod
    @property
    def config(self) -> dict:
        """ 获取过滤策略配置
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, examples: list) -> list:
        """ 过滤示例
        """
        raise NotImplementedError


class F1Filter(Filter):
    def __init__(self,
                 llm: LLM,
                 prompt: Prompt,
                 f1_func: Callable[[dict, str], float],
                 filter_strategy: Literal['percentage', 'fixed_quantity'] = 'percentage',
                 **kwargs
                 ) -> None:
        """ F1 值过滤策略
        
        Args:
            llm (LLM): 语言模型
            prompt (Prompt): 提示词
            f1_func (Callable[[dict, str], float]): F1 值计算函数, 输入是当前数据和模型预测结果, 输出是 F1 值
            filter_strategy (Literal['percentage', 'fixed_quantity']): 过滤策略
            **kwargs: 其他参数
        """
        self.llm = llm
        self.prompt = prompt
        self.f1_func = f1_func
        self.filter_strategy = filter_strategy
        self.kwargs = kwargs

    @property
    def config(self) -> dict:
        """ 获取过滤策略配置
        """
        return {
            'llm': self.llm.config,
            'filter_strategy': self.filter_strategy,
            'kwargs': self.kwargs
        }

    def filter(self, examples: list) -> list:
        """ 过滤示例
        
        Args:
            examples (list): 示例列表

        Returns:
            list: 过滤后的示例列表
        """
        data = []
        for example in examples:
            prompt, instruction = self.prompt.get_ner_prompt(example['text'])
            self.llm.instruction = instruction
            resp, _ = self.llm.chat(prompt)
            f1 = self.f1_func(example, resp)
            example['f1'] = f1
            data.append(example)
        match self.filter_strategy:
            case 'percentage':
                data.sort(key=lambda x: x['f1'], reverse=True)
                return data[:int(len(data) * self.kwargs.get('filter_percent', 0.4))]
            case 'fixed_quantity':
                return data[:self.kwargs.get('filter_quantity', 100)]
            case _:
                return data
