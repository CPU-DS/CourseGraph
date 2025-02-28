# -*- coding: utf-8 -*-
# Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File: course_graph/database/mongo.py
# Description: 定义 mongodb 数据库连接类

from pymongo import MongoClient
from pymongo.collection import Collection
from singleton_decorator import singleton


@singleton
class Mongo:

    def __init__(self, url: str, db: str):
        """ 连接到 Mongodb 数据库

        Args:
            url (str): 地址
            db (str): 数据库
        """
        self.__client = MongoClient(url)
        self.__db = self.__client[db]

    def get_collection(self, collection: str) -> Collection:
        """ 获取数据库集合

        Args:
            collection (str): 集合名称

        Returns:
            Collection: 集合
        """
        return self.__db[collection]
