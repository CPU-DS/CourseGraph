# -*- coding: utf-8 -*-
# Create Date: 2024/07/13
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/prompt_strategy.py
# Description: 定义提示词示例检索策略

from tqdm import tqdm
import numpy as np
from pymilvus import MilvusClient
from ..llm import LLM


class SentenceEmbeddingStrategy:

    def __init__(self,
                 embed_model: LLM,
                 milvus_path: str = 'src/course_graph/database/milvus.db',
                 topk: int = 3,
                 avoid_first: bool = False) -> None:
        """ 基于句嵌入相似度的示例检索策略

        Args:
            embed_model (LLM): 嵌入模型
            milvus_path (str, optional): 向量数据库 milvus 存储地址. Defaults to 'src/course_graph/database/milvus.db'.
            topk (int, optional): 选择排名前topk个示例. Defaults to 3.
            avoid_first (bool, optional): 去掉相似度最大的那个示例且不减少最终topk数量. Default to False.
        """
        super().__init__()
        self.client = MilvusClient(milvus_path)
        self.embed_model = embed_model
        self.collection = 'prompt_example'
        self.topk = topk
        self.avoid_first = avoid_first

        if self.avoid_first:
            self.topk += 1

    def reimport_example(
            self,
            embed_dim: int,
            data: list
        ) -> None:
        """ 重新向数据库中导入提示词示例

        Args:
            embed_dim (int): 嵌入维度
            data_path (str, optional): 源数据地址
        """

        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            
        self.client.create_collection(
            collection_name=self.collection,
            dimension=embed_dim
        )
        examples = []
        for idx, line in enumerate(data):
            line['id'] = idx
            examples.append(line)

        for idx in tqdm(range(len(examples))):
            examples[idx]['vector'] = np.array(self.embed_model.embedding(
                input=examples[idx]['text'],
                dimensions=embed_dim
            )).astype('float32')
        self.client.insert(
            collection_name=self.collection,
            data=examples
        )

    def _get_example_by_sts_similarity(self, content: str) -> list:
        """ 使用待抽取内容content和库中已有文本片段text的句相似度进行example检索

        Args:
            content (str): 待抽取的文本内容
            collection (str): 集合名称

        Returns:
            list: 提示词示例列表

        """
        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
        content_vec = self.embed_model.encode(content,
                                              normalize_embeddings=True)
        resp = self.client.search(
            collection_name=self.collection,
            data=np.array([content_vec]).astype('float32'),
            limit=self.topk if not self.avoid_first else self.topk + 1,
            output_fields=["text"]
        )
        resp = [i['entity']['text'] for i in resp[0]]
        if self.avoid_first:
            resp = resp[1:]
        return resp

    def get_example(self, content: str) -> list:
        """ 获取示例

        Args:
            content (str): 文本内容

        Returns:
            list: 原始数据
        """
        return self._get_example_by_sts_similarity(content)
