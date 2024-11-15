# -*- coding: utf-8 -*-
# Create Date: 2024/07/13
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt_strategy.py
# Description: 定义提示词示例检索策略

from ..database import Mongo, Faiss
import os
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from glob import glob
from typing import Literal
from abc import ABC, abstractmethod
from .type import Database


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
                 mongo_url: str = 'mongodb://localhost:27017/',
                 faiss_path: str = 'course_graph/database/faiss_index',
                 topk: int = 3,
                 avoid_first: bool = False) -> None:
        """ 基于句嵌入相似度的示例检索策略

        Args:
            embed_model_path (str): 嵌入模型路径
            mongo_url (str, optional): 文档数据库 mongodb 地址. Defaults to 'mongodb://localhost:27017/'.
            faiss_path (str, optional): 向量数据库 faiss 存储地址. Defaults to 'course_graph/database/faiss_index'.
            topk (int, optional): 选择排名前topk个示例. Defaults to 3.
            avoid_first (bool, optional): 去掉相似度最大的那个示例且不减少最终topk数量. Default to False.
        """
        super().__init__()
        mongo = Mongo(mongo_url, 'course_graph')
        self.db_ner = Database(
            faiss=Faiss(os.path.join(faiss_path, 'faiss_index_ner.bin')),
            mongo=mongo.get_collection('prompt_example_ner'))
        self.db_re = Database(faiss=Faiss(
            os.path.join(faiss_path, 'faiss_index_re.bin')),
                              mongo=mongo.get_collection('prompt_example_re'))
        self.db_ae = Database(faiss=Faiss(
            os.path.join(faiss_path, 'faiss_index_ae.bin')),
                              mongo=mongo.get_collection('prompt_example_ae'))
        self.embed_model = SentenceTransformer(embed_model_path)
        self.topk = topk
        self.avoid_first = avoid_first

        if self.avoid_first:
            self.topk += 1

    def reimport_example(
            self,
            embed_dim: int,
            example_dataset_path: str = 'dataset/prompt_example') -> None:
        """ 重新向数据库(文档数据库/向量数据库)中导入提示词示例

        Args:
            embed_dim (int): 嵌入维度
            example_dataset_path (str, optional): 提示词示例源数据地址文件夹. Defaults to 'dataset/prompt_example'.
        """

        for file in glob(example_dataset_path + '/*'):
            if file.endswith('ner.json'):
                text_name = "input"  # 实体抽取中 模型输入input和文本片段text相同
                db = self.db_ner
            elif file.endswith('re.json'):
                text_name = "text"  # 关系/属性抽取中 存入向量库中的应当是文本片段text
                db = self.db_re
            elif file.endswith('ae.json'):
                text_name = "text"
                db = self.db_ae
            else:
                continue
            # 清除文档数据库中已保存内容
            db.mongo.drop()
            examples = []
            idx = 0
            with open(file, 'r', encoding='UTF-8') as f:
                for line in json.load(f):
                    line['index'] = idx
                    idx += 1
                    examples.append(line)
            db.mongo.insert_many(examples)
            data = []
            for line in examples:
                data.append(
                    self.embed_model.encode(line[text_name],
                                            normalize_embeddings=True))
            # 清除向量数据库中已保存内容
            db.faiss.delete()
            db.faiss.create(embed_dim).add(
                np.array(data).astype('float32')).save()

    def _get_example_by_sts_similarity(
            self, content: str, type_: Literal["ner", "re", "ae"]) -> list:
        """ 使用待抽取内容content和库中已有文本片段text的句相似度进行example检索

        Args:
            content (str): 待抽取的文本内容
            type_((Literal["ner", "re", "ae"])): 实体/关系/属性 抽取

        Returns:
            list: 提示词示例列表

        """
        match type_:
            case "ner":
                db = self.db_ner
            case "re":
                db = self.db_re
            case _:
                db = self.db_ae
        if db.faiss.index is None:
            db.faiss.load()
        content_vec = self.embed_model.encode(content,
                                              normalize_embeddings=True)
        idx = db.faiss.search(
            np.array([content_vec]).astype('float32'), self.topk)
        examples = []
        for i in idx:
            res = db.mongo.find_one({'index': i})
            examples.append({"input": res["input"], 'output': res["output"]})
        if self.avoid_first:
            examples = examples[1:]
        return examples

    def get_ner_example(self, content: str) -> list:
        """ 获取实体抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        return self._get_example_by_sts_similarity(content, "ner")

    def get_re_example(self, content: str) -> list:
        """ 获取关系抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        return self._get_example_by_sts_similarity(content, "re")

    def get_ae_example(self, content: str) -> list:
        """ 获取属性抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        return self._get_example_by_sts_similarity(content, "ae")
