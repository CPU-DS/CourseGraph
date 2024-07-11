# -*- coding: utf-8 -*-
# Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File: coursekg/database/neo4j.py
# Description: 定义图数据库连接类

from py2neo import Graph
from tqdm import tqdm


def singleton(cls):

    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class Neo4j:

    def __init__(self, url: str, username: str, password: str) -> None:
        """ 连接到 Neo4j 数据库

        Args:
            url (str): 地址
            username (str): 用户名
            password (str): 密码
        """
        self.graph = Graph(url, auth=(username, password), name='neo4j')

    def run(self, cyphers: str | list[str]):
        """ 执行一条或多条 cypher 语句

        Args:
            cyphers (str | list[str]): 一条或多条cypher语句
        """
        if isinstance(cyphers, str):
            return self.graph.run(cyphers)
        else:
            res = []
            for cypher in tqdm(cyphers):
                res.append(self.graph.run(cypher))
            return res
