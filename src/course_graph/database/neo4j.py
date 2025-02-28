# -*- coding: utf-8 -*-
# Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File: course_graph/database/neo4j.py
# Description: 定义图数据库连接

from py2neo import Graph
from tqdm import tqdm
from typing import overload
from singleton_decorator import singleton

PROTOCOLS = ['bolt', 'http', 'https', 'neo4j']
DEFAULT_PROTOCOL = PROTOCOLS[0]

@singleton
class Neo4j(Graph):

    def __init__(self, url: str, username: str, password: str, graph_name = 'neo4j') -> None:
        """ 连接到 Neo4j 数据库

        Args:
            url (str): 连接地址
            username (str): 用户名
            password (str): 密码
            graph_name (str, optional): 图数据库名称. Defaults to 'neo4j'.
        """
        if not any(url.startswith(protocol) for protocol in PROTOCOLS):
            url = f'{DEFAULT_PROTOCOL}://{url}'
        super().__init__(url, auth=(username, password), name=graph_name)
        
    @overload
    def run(self, cyphers: str | list[str]):
        """ 执行一条或多条 cypher 语句

        Args:
            cyphers (str | list[str]): 一条或多条cypher语句
        """
        if isinstance(cyphers, str):
            return self.graph.run(cyphers)
        else:
            res = []
            for cypher in tqdm(cyphers, desc='执行 cypher 语句'):
                res.append(self.graph.run(cypher))
            return res
