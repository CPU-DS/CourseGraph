# -*- coding: utf-8 -*-
# Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File: course_graph/database/neo4j.py
# Description: 定义图数据库连接

from py2neo import Graph, Node, Relationship
from singleton_decorator import singleton
from functools import lru_cache

PROTOCOLS = ['bolt', 'http', 'https', 'neo4j']
DEFAULT_PROTOCOL = PROTOCOLS[0]


@singleton
class Neo4j(Graph):

    def __init__(self, url: str, username: str, password: str, graph_name='neo4j') -> None:
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

    @lru_cache
    def match_nodes(self, skip: int = None, limit: int = None) -> list[Node]:
        """ 获取所有 Node
        
        Args:
            skip (int, optional): 跳过. Defaults to None.
            limit (int, optional): 限制. Defaults to None.

        Returns:
            list: 所有 Node
        """
        query = self.nodes.match()
        if skip is not None:
            query = query.skip(skip)
        if limit is not None:
            query = query.limit(limit)
        return query.all()
    
    @lru_cache
    def match_relations(self, skip: int = None, limit: int = None) -> list[Relationship]:
        """ 获取所有 Relation
        
        Args:
            skip (int, optional): 跳过. Defaults to None.
            limit (int, optional): 限制. Defaults to None.

        Returns:
            list: 所有 Relation
        """
        query = self.relationships.match()
        if skip is not None:
            query = query.skip(skip)
        if limit is not None:
            query = query.limit(limit)
        return query.all()
