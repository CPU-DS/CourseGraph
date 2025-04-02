# -*- coding: utf-8 -*-
# Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File: course_graph/database/neo4j.py
# Description: 定义图数据库连接

from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship
from singleton_decorator import singleton
from functools import cache


@singleton
class Neo4j:

    def __init__(self, url: str, username: str, password: str, graph_name: str = 'neo4j'):
        """ 连接到 Neo4j 数据库

        Args:
            url (str): 连接地址
            username (str): 用户名
            password (str): 密码
            graph_name (str, optional): 数据库名称. Defaults to 'neo4j'.
        """
        self.url = url
        self.graph_name = graph_name
        self.driver = GraphDatabase.driver(url, auth=(username, password))
        self.driver.verify_connectivity()
        self.session = self.driver.session(database=graph_name)
    
    def __enter__(self) -> 'Neo4j':
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def close(self):
        """ 关闭数据库连接 """
        self.session.close()
        self.driver.close()

    @cache
    def get_nodes(self, skip: int = None, limit: int = None) -> list[Node]:
        """ 获取所有 Node
        
        Args:
            skip (int, optional): 跳过. Defaults to None.
            limit (int, optional): 限制. Defaults to None.

        Returns:
            list: 所有 Node
        """
        query = "MATCH (n) RETURN n"
        if skip is not None:
            query += f" skip $skip"
        if limit is not None:
            query += f" limit $limit"
        records, _, _ = self.driver.execute_query(query, limit=limit, skip=skip)
        return [record['n'] for record in records]
    
    @cache
    def get_relations(self, skip: int = None, limit: int = None) -> list[Relationship]:
        """ 获取所有 Relation
        
        Args:
            skip (int, optional): 跳过. Defaults to None.
            limit (int, optional): 限制. Defaults to None.

        Returns:
            list: 所有 Relation
        """
        query = "MATCH (m)-[r]->(n) RETURN m, r, n"
        if skip is not None:
            query += f" skip $skip"
        if limit is not None:
            query += f" limit $limit"
        records, _, _ = self.driver.execute_query(query, limit=limit, skip=skip)
        return [record['r'] for record in records]
    
    @cache
    def get_nodes_count(self) -> int:
        """ 获取所有 Node 的数量
        
        Returns:
            int: 所有 Node 的数量
        """
        records, _, _ = self.driver.execute_query(
            "MATCH (n) RETURN count(n)"
        )
        return records[0]['count(n)']
    
    @cache
    def get_relations_count(self) -> int:
        """ 获取所有 Relation 的数量
        
        Returns:
            int: 所有 Relation 的数量
        """
        records, _, _ = self.driver.execute_query(
            "MATCH ()-[r]->() RETURN count(r)"
        )
        return records[0]['count(r)']
    
    @cache
    def get_max_relation_count(self) -> int:
        """ 获取所有 Relation 的最大 ID
        
        Returns:
            int: 所有 Relation 的最大 ID
        """
        records, _, _ = self.driver.execute_query("""
            MATCH (n)-[r]-()
            RETURN n, count(r) AS relation_count
            ORDER BY relation_count DESC
            LIMIT 1""")
        return records[0]['relation_count']
    
    @cache
    def get_nodes_with_relation_count(self, skip: int = None, limit: int = None) -> list[tuple[Node, int]]:
        """ 获取所有 Node 及其关系数量
        
        Args:
            skip (int, optional): 跳过. Defaults to None.
            limit (int, optional): 限制. Defaults to None.

        Returns:
            list[tuple[Node, int]]: 所有 Node 及其关系数量
        """
        query = "MATCH (n)-[r]-() RETURN n, count(r)"
        if skip is not None:
            query += f" skip $skip"
        if limit is not None:
            query += f" limit $limit"
        records, _, _ = self.driver.execute_query(query, limit=limit, skip=skip)
        return [(record['n'], record['count(r)']) for record in records]
    
    @cache
    def get_node_by_id(self, id: int) -> Node:
        """ 获取指定 id 的 Node
        
        Args:
            id (int): 指定 id
            
        Returns:
            Node: 相应 Node
        """
        records, _, _ = self.driver.execute_query("MATCH (n) WHERE n.id = $id RETURN n", id=id)
        return records[0]['n']
    
    @cache
    def get_relations_by_node_id(self, id: int) -> list[Relationship]:
        """ 获取指定 id 的 Node 的所有 Relation
        
        Args:
            id (int): 指定 id
            
        Returns:
            list[Relationship]: 相应 Node 的所有 Relation
        """
        records, _, _ = self.driver.execute_query("MATCH (n)-[r]->() WHERE n.id = $id RETURN r", id=id)
        return [record['r'] for record in records]
    
    def __hash__(self):
        return hash(self.url)
    