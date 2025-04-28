# -*- coding: utf-8 -*-
# Create Date: 2025/04/02
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/core.py
# Description: 使用大模型抽取知识点

from ..llm.prompt import ExamplePrompt, Prompt, post_process
from ..llm import LLM
import random
from collections import Counter
from loguru import logger


def get_knowledgepoint_entities_by_llm(
        content: str,
        llm: LLM,
        prompt: Prompt = ExamplePrompt(),
        self_consistency: bool = False,
        samples: int = 5,
        top: float = 0.5
) -> dict:
    """ 使用大模型抽取知识点

    Args:
        content (str): 文本内容
        llm (LLM): 大模型
        prompt (Prompt, optional): 提示词生成器. Defaults to ExamplePrompt().
        self_consistency (bool, optional): 是否使用自一致性策略. Defaults to False.
        samples (int, optional): 采样次数. Defaults to 5.
        top (float, optional): 置信度阈值. Defaults to 0.5.

    Returns:
        dict: 知识点列表
    """
    message, instruction = prompt.get_ner_prompt(content)
    llm.instruction = instruction
    if not self_consistency:
        # 默认策略：实体生成数量过多则重试，否则随机选择5个
        retry = 0
        while True:
            resp, _ = llm.chat(message)
            entities: dict = post_process(resp) or {}
            if all(len(value) < 8 for value in entities.values()) or retry >= 3:
                break
            retry += 1
        for entity_type, entity_list in entities.items():
            if len(entity_list) > 10:
                entities[entity_type] = random.sample(entity_list, 5)
    else:
        # 自我一致性验证
        all_entities: list[dict] = []
        for idx in range(samples):
            resp, _ = llm.chat(message)
            logger.info(f'第{idx}次采样: ' + resp)
            entities: dict = post_process(resp) or {}

            logger.info(f'获取知识点实体: ' + str(entities))
            all_entities.append(entities)

        entities = {}
        # 这里的自我一致性是要求每种类型中提及的实体超过一定数量
        for entity_type in {k for d in all_entities for k in d}:  # 所有的 keys
            elements = [item for d in all_entities if entity_type in d for item in d[entity_type]]
            entities[entity_type] = [point for point, count in Counter(elements).items() if count > (samples * top)]
    # 分支结束得到 entities {'entity_type': ['entity1', 'entity2']}
    return entities


def get_knowledgepoint_attributes_by_llm(
        content: str,
        knowledgepoints: list[str],
        llm: LLM,
        prompt: Prompt = ExamplePrompt()
) -> dict:
    """ 使用大模型抽取知识点属性

    Args:
        content (str): 文本内容
        knowledgepoints (list[str]): 知识点列表
        llm (LLM): 大模型
        prompt (Prompt, optional): 提示词生成器. Defaults to ExamplePrompt().
    """
    message, instruction = prompt.get_ae_prompt(content, knowledgepoints)
    llm.instruction = instruction
    resp, _ = llm.chat(message)
    attrs: dict = post_process(resp) or {}
    # attrs {'entity1': {'attribute1': 'value1', 'attribute2': 'value2'}}
    return attrs


def get_knowledgepoint_relations_by_llm(
        content: str,
        knowledgepoints: list[str],
        llm: LLM,
        prompt: Prompt = ExamplePrompt(),
        self_consistency: bool = False,
        samples: int = 5,
        top: float = 0.5
) -> dict:
    """ 使用大模型抽取知识点关系

    Args:
        content (str): 文本内容
        knowledgepoints (list[str]): 知识点列表
        llm (LLM): 大模型
        prompt (Prompt, optional): 提示词生成器. Defaults to ExamplePrompt().
        self_consistency (bool, optional): 是否使用自一致性策略. Defaults to False.
        samples (int, optional): 采样次数. Defaults to 5.
        top (float, optional): 置信度阈值. Defaults to 0.5.

    Returns:
        dict: 关系三元组列表
    """
    message, instruction = prompt.get_re_prompt(content, knowledgepoints)
    llm.instruction = instruction
    if not self_consistency:
        resp, _ = llm.chat(message)
        relations = post_process(resp) or []
    else:
        all_relations = []
        for idx in range(samples):
            resp, _ = llm.chat(message)
            logger.info(f'第{idx}次采样: ' + resp)
            relations = post_process(resp) or []
            all_relations.extend(relations)
        relations = [
            dict(relation)
            for relation, count in Counter(frozenset(relation.items()) for relation in all_relations).items() if
            count > (samples * top)
        ]
    # 分支结束得到 relations [{'head':'', 'relation':'', 'tail':''}]
    return relations


def get_knowledgepoint_attribute_only_by_llm(
        knowledgepoint: str,
        attribute: str,
        value_list: list[str],
        llm: LLM,
        prompt: Prompt = ExamplePrompt()
) -> str:
    """ 使用大模型总结知识点属性

    Args:
        knowledgepoint (str): 知识点
        attribute (str): 属性
        value_list (list[str]): 属性值列表
        llm (LLM): 大模型
        prompt (Prompt, optional): 提示词生成器. Defaults to ExamplePrompt().
    
    Returns:
        str: 属性值
    """
    message, instruction = prompt.get_best_attr_prompt(knowledgepoint, attribute, value_list)
    llm.instruction = instruction
    resp, _ = llm.chat(message)
    return resp
