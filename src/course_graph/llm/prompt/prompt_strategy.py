# -*- coding: utf-8 -*-
# Create Date: 2024/07/13
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/prompt_strategy.py
# Description: 定义提示词示例检索策略

from tqdm import tqdm
import numpy as np
from pymilvus import MilvusClient
from ..llm import LLM
from abc import ABC, abstractmethod


class PromptStrategy(ABC):
    """ 提示词策略
    """
    
    @property
    @abstractmethod
    def config(self) -> dict:
        """ 获取策略配置
        """
        raise NotImplementedError

    @abstractmethod
    def get_ner_example(self, content: str, filter: str = None) -> list:
        """ 获取实体抽取示例
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_re_example(self, content: str, entities: list) -> list:
        """ 获取关系抽取示例
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_ae_example(self, content: str, entities: list) -> list:
        """ 获取属性抽取示例
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_best_attr_example(self, entity: str, attr: str, values: list) -> list:
        """ 获取属性抽取示例
        """
        raise NotImplementedError


class SentenceEmbeddingStrategy(PromptStrategy):

    def __init__(self,
                 embed_model: LLM,
                 milvus_path: str = 'src/course_graph/database/milvus.db',
                 topk: int = 3,
                 embed_dim: int = 768,
                 avoid_first: bool = False) -> None:
        """ 基于句嵌入相似度的示例检索策略

        Args:
            embed_model (LLM): 嵌入模型
            milvus_path (str, optional): 向量数据库 milvus 存储地址. Defaults to 'src/course_graph/database/milvus.db'.
            topk (int, optional): 选择排名前 topk 个示例. Defaults to 3.
            embed_dim (int, optional): 嵌入维度. Defaults to 768.
            avoid_first (bool, optional): 去掉相似度最大的那个示例且不减少最终 topk 数量. Default to False.
        """
        super().__init__()
        self.client = MilvusClient(milvus_path)
        self.embed_model = embed_model
        self.collection = 'prompt_example'
        self.topk = topk
        self.avoid_first = avoid_first
        self.embed_dim = embed_dim

        self.json_block: bool = True
            
    @property
    def config(self) -> dict:
        return {
            'topk': self.topk,
            'avoid_first': self.avoid_first,
            'embed_dim': self.embed_dim,
            'json_block': self.json_block
        }

    def reimport_example(
            self,
            data: list) -> None:
        """ 重新向数据库中导入提示词示例

        Args:
            data (list): 源数据, 每一项需要包含 `text` 字段
        """

        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.embed_dim
        )
        examples: list[dict] = []
        for idx, line in enumerate(data):
            line['id'] = idx
            examples.append(line)

        for idx in tqdm(range(len(examples))):
            examples[idx]['vector'] = np.array(self.embed_model.embedding(
                input=examples[idx]['text'],
                dimensions=self.embed_dim
            )).astype('float32')
        self.client.insert(
            collection_name=self.collection,
            data=examples
        )

    def _get_example_by_sts_similarity(self, content: str, filter: str = None) -> list:
        """ 使用待抽取内容 content 和库中已有文本片段 text 的句相似度进行 example 检索

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表

        """
        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
        content_vec = self.embed_model.embedding(
            input=content,
            dimensions=self.embed_dim
        )
        resp = self.client.search(
            collection_name=self.collection,
            data=np.array([content_vec]).astype('float32'),
            limit=self.topk if not self.avoid_first else self.topk + 1,
            search_params={
                'metric_type': 'COSINE'
            },
            filter=filter if filter else '',
            output_fields=['*']
        )
        resp = [i['entity'] for i in resp[0]]
        if self.avoid_first:
            resp = resp[1:]
        return resp

    def get_ner_example(self, content: str, filter: str = None) -> list:
        """ 获取命名实体识别示例

        Args:
            content (str): 待抽取的文本内容
            filter (str, optional): 过滤条件. Defaults to None.

        Returns:
            list: 提示词示例列表
        """
        examples = []
        for example in self._get_example_by_sts_similarity(content, filter):  # topk 个示例
            output = {}
            for e in example['entities']:
                type_ = e['type']
                if type_ not in output:
                    output[type_] = []
                output[type_].append(e['text'])  # 添加相应类的实体
            output = f'{output}'
            if self.json_block:
                output = f'```json\n{output}\n```'
            examples.append({
                '输入': example['text'],
                '输出': output
            })
        return examples
    
    def get_re_example(self, content: str, entities: list) -> list:
        return []
    
    def get_ae_example(self, content: str, entities: list) -> list:
        return []
    
    def get_best_attr_example(self, entity: str, attr: str, values: list) -> list:
        return []
