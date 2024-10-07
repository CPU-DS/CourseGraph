# -*- coding: utf-8 -*-
# Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File: course_graph/database/vector.py
# Description: 定义向量数据库连接类

import faiss
import os
import numpy as np


class Faiss:

    def __init__(self, index_path: str) -> None:
        """ 使用 faiss 向量数据库
        
        Args:
            index_path (str, optional): faiss 向量数据库索引文件地址.
        """
        self.index_path = index_path
        self.index: faiss.IndexFlatL2 | None = None

    def delete(self) -> None:
        """ 删除索引
        """
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    def load(self) -> 'Faiss':
        """ 加载索引
        """
        self.index = faiss.read_index(self.index_path)
        return self

    def search(self, vec: np.ndarray, k: int) -> list[int]:
        """ 查询

        Args:
            vec (np.ndarray): 查询向量
            k (int): 查询返回前k个

        Returns:
            list[int]: 前k个值插入时位置
        """
        if self.index:
            _, idx = self.index.search(vec, k)
            return [int(i) for i in idx[0]]
        else:
            raise ValueError("请先加载或创建索引")

    def create(self, embed_dim: int) -> 'Faiss':
        """ 新建索引

        Args:
            embed_dim (int): 嵌入维度
        """
        self.index = faiss.IndexFlatL2(embed_dim)
        return self

    def add(self, data: np.ndarray) -> 'Faiss':
        """ 插入数据

        Args:
            data (np.ndarray): 数据
        """
        self.index.add(data)
        return self

    def save(self) -> None:
        """ 持久化索引
        """
        if self.index:
            faiss.write_index(self.index, self.index_path)
        else:
            raise ValueError("请先加载或创建索引")
