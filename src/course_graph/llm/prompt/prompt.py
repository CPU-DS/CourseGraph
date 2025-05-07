# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/prompt.py
# Description: 定义提示词类

from abc import ABC, abstractmethod
import json
from ..ontology import ONTOLOGY
from typing import Literal
from .utils import json2md
from .prompt_strategy import PromptStrategy


class Prompt(ABC):

    def __init__(self) -> None:
        """ 信息抽取提示词类
        """
        pass

    @abstractmethod
    def get_ner_prompt(self, content: str) -> tuple[str, str]:
        """ 获取实体抽取提示词
        """
        raise NotImplementedError

    @abstractmethod
    def get_re_prompt(self, content: str,
                      entities: list[str]) -> tuple[str, str]:
        """ 获取关系抽取的提示词
        """
        raise NotImplementedError

    @abstractmethod
    def get_ae_prompt(self, content: str,
                      entities: list[str]) -> tuple[str, str]:
        """ 获取属性抽取的提示词
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_attr_prompt(self, entity: str, attr: str,
                             values: list[str]) -> tuple[str, str]:
        """ 要求模型为实体的属性总结一个最佳的值
        """
        raise NotImplementedError


class ExamplePrompt(Prompt):

    def __init__(self, type: Literal['json', 'md'] = 'json', strategy: PromptStrategy = None) -> None:
        """ 获取提取提示词, 使用多种提示词优化, 包括CoT、基于动态检索的ICL

        Args:
            type (Literal['json', 'md'], optional): 提示词格式. Defaults to 'json'.
        """
        super().__init__()
        self.type = type
        self.strategy = strategy
    
    def get_ner_prompt(self, 
                       content: str) -> tuple[str, str]:
        examples = []
        if self.strategy is not None:
            examples = self.strategy.get_ner_example(content)
        prompt = {
            "任务":
                "请对输入的内容进行总结根据总结从中抽取出符合schema类型的实体。最后请给出你的总结和抽取到的类型以及对应的列表, 返回的格式为\n```json\n{\"entity_type1\": [\"entity1\", \"entity2\"]}\n```",
            "schema": ONTOLOGY['entities'],
            "示例": examples,
            "输入": content
        }
        resp = json.dumps(prompt, indent=4, ensure_ascii=False) if self.type == 'json' else json2md(prompt)
        return resp, "你是专门进行知识点实体抽取的专家"

    def get_re_prompt(self, 
                      content: str,
                      entities: list[str]) -> tuple[str, str]:
        examples = []
        if self.strategy is not None:
            examples = self.strategy.get_re_example(content, entities)
        prompt = {
            "任务":
                "请根据提供的中心知识点和已有文本片段, 一步步思考, 寻找与之相关联的知识点并判断二者之间的关系, 如果存在关系但不在所指定的关系范围relations中, 则不返回。头尾实体不应该相同。返回为你的思考和关系三元组, 格式为\n```json\n[{\"head\": \"\", \"relation\": \"\", \"tail\": \"\"}]\n```",
            "relations": ONTOLOGY['relations'],
            "示例": examples,
            "输入": f"中心实体列表为: {entities}, 文本片段为: '{content}'"
        }
        resp = json.dumps(prompt, indent=4, ensure_ascii=False) if self.type == 'json' else json2md(prompt)
        return resp, "你是专门进行知识点关系判别的专家"

    def get_ae_prompt(self, 
                      content: str,
                      entities: list[str]) -> tuple[str, str]:
        examples = []
        if self.strategy is not None:
            examples = self.strategy.get_ae_example(content, entities)
        prompt = {
            "任务":
                "请对输入的实体列表根据已有文本片段各自抽取他们的属性值。属性范围只能来源于提供的attributes, 属性值无需完全重复原文, 可以是你根据原文进行的总结, 如果实体没有能够总结的属性值则不返回。返回格式为\n```json\n{\"entity1\": {\"attribute1\":\"value\"}}\n```",
            "attributes": ONTOLOGY['attributes'],
            "示例": examples,
            "输入": f"实体列表为: {entities}, 文本片段为: '{content}'"
        }
        resp = json.dumps(prompt, indent=4, ensure_ascii=False) if self.type == 'json' else json2md(prompt)
        return resp, "你是专门进行知识点属性抽取的专家"

    def get_best_attr_prompt(self, 
                             entity: str, 
                             attr: str,
                             values: list[str]) -> tuple[str, str]:
        examples = []
        if self.strategy is not None:
            examples = self.strategy.get_best_attr_example(entity, attr, values)
        prompt = {
            "任务":
                "请根据实体的属性对应的值列表, 总结出一个最佳的属性值。只需要返回总结的属性值即可。",
            "示例": examples,
            "输入": f"实体为: '{entity}', 属性为: '{attr}', 属性值列表为: {values}"
        }
        resp = json.dumps(prompt, indent=4, ensure_ascii=False) if self.type == 'json' else json2md(prompt)
        return resp, "你是专门进行属性判别的专家"
