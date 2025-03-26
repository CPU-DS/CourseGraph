# -*- coding: utf-8 -*-
# Create Date: 2024/07/13
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt_strategy.py
# Description: 定义提示词示例检索策略

import os
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from glob import glob
from typing import Literal
from abc import ABC, abstractmethod
from pymilvus import MilvusClient


class ExamplePromptStrategy(ABC):

    def __init__(self):
        """ 提示词示例检索策略
        """
        pass

    @abstractmethod
    def get_ner_example(self, content: str) -> list:
        """ 获取实体抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list: 提示词示例列表
        """
        raise NotImplementedError

    @abstractmethod
    def get_re_example(self, content: str) -> list:
        """ 获取关系抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list: 提示词示例列表
        """
        raise NotImplementedError

    @abstractmethod
    def get_ae_example(self, content: str) -> list:
        """ 获取属性抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list: 提示词示例列表
        """
        raise NotImplementedError


class SentenceEmbeddingStrategy(ExamplePromptStrategy):

    def __init__(self,
                 embed_model_path: str,
                 milvus_path: str = 'src/course_graph/database/milvus.db',
                 topk: int = 3,
                 avoid_first: bool = False) -> None:
        """ 基于句嵌入相似度的示例检索策略

        Args:
            embed_model_path (str): 嵌入模型路径
            milvus_path (str, optional): 向量数据库 milvus 存储地址. Defaults to 'src/course_graph/database/milvus.db'.
            topk (int, optional): 选择排名前topk个示例. Defaults to 3.
            avoid_first (bool, optional): 去掉相似度最大的那个示例且不减少最终topk数量. Default to False.
        """
        super().__init__()
        self.client = MilvusClient(milvus_path)
        self.ner_collection = 'prompt_example_ner'
        self.re_collection = 'prompt_example_re'
        self.ae_collection = 'prompt_example_ae'
        self.embed_model = SentenceTransformer(embed_model_path)
        self.topk = topk
        self.avoid_first = avoid_first

        if self.avoid_first:
            self.topk += 1

    def reimport_example(
            self,
            embed_dim: int,
            example_dataset_path: str = 'dataset/prompt_example') -> None:
        """ 重新向数据库中导入提示词示例

        Args:
            embed_dim (int): 嵌入维度
            example_dataset_path (str, optional): 提示词示例源数据地址文件夹. Defaults to 'dataset/prompt_example'.
        """

        for file in glob(example_dataset_path + '/*'):
            if file.endswith('ner.json'):
                text_name = "input"  # 实体抽取中 模型输入input和文本片段text相同
                collection = self.ner_collection
            elif file.endswith('re.json'):
                text_name = "text"  # 关系/属性抽取中 存入向量库中的应当是文本片段text
                collection = self.re_collection
            elif file.endswith('ae.json'):
                text_name = "text"
                collection = self.ae_collection
            else:
                continue

            if self.client.has_collection(collection):
                self.client.drop_collection(collection)
            
            self.client.create_collection(
                collection_name=collection,
                dimension=embed_dim
            )
            examples = []
            with open(file, 'r', encoding='UTF-8') as f:
                for idx, line in enumerate(json.load(f)):
                    line['id'] = idx
                    line['text'] = line[text_name]  # 统一替换为 text 字段
                    del line[text_name]
                    examples.append(line)
    
            for idx in range(len(examples)):
                examples[idx]['vector'] = np.array(self.embed_model.encode(examples[idx]['text'],
                                            normalize_embeddings=True)).astype('float32')
            self.client.insert(
                collection_name=collection,
                data=examples
            )

    def _get_example_by_sts_similarity(
            self, content: str, collection: str) -> list:
        """ 使用待抽取内容content和库中已有文本片段text的句相似度进行example检索

        Args:
            content (str): 待抽取的文本内容
            collection (str): 集合名称

        Returns:
            list: 提示词示例列表

        """
        if self.client.has_collection(collection):
            self.client.load_collection(collection)
        content_vec = self.embed_model.encode(content,
                                              normalize_embeddings=True)
        resp = self.client.search(
            collection_name=collection,
            data=np.array([content_vec]).astype('float32'),
            limit=self.topk if not self.avoid_first else self.topk + 1,
            output_fields=["text"]
        )
        resp = [i['entity']['text'] for i in resp[0]]
        if self.avoid_first:
            resp = resp[1:]
        return resp

    def get_ner_example(self, content: str) -> list:
        """ 获取实体抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        return self._get_example_by_sts_similarity(content, self.ner_collection)

    def get_re_example(self, content: str) -> list:
        """ 获取关系抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        return self._get_example_by_sts_similarity(content, self.re_collection)

    def get_ae_example(self, content: str) -> list:
        """ 获取属性抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        return self._get_example_by_sts_similarity(content, self.ae_collection)
