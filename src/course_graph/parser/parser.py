# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/parser.py
# Description: 定义文档解析器基类

from __future__ import annotations
from abc import ABC, abstractmethod
from .document import Document
from .types import BookMark, Content


class Parser(ABC):

    def __init__(self, file_path: str) -> None:
        """ 文档解析器基类

        Args:
            file_path (str): 文档路径
        """
        super().__init__()
        self.file_path = file_path

    def __enter__(self) -> 'Parser':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @abstractmethod
    def close(self) -> None:
        """ 关闭文档

        Raises:
            NotImplementedError: 子类需要实现该方法
        """
        raise NotImplementedError

    @abstractmethod
    def get_bookmarks(self) -> list[BookMark]:
        """  获取pdf文档书签

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list[BookMark]: 书签列表
        """
        raise NotImplementedError

    @abstractmethod
    def get_contents(self, bookmark: BookMark) -> list[Content]:
        """  获取书签下的所有内容

        Args:
            bookmark (BookMark): 书签

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list[Content]: 内容列表
        """
        raise NotImplementedError

    def get_document(self) -> Document:
        """ 获取文档

        Returns:
            Document: 文档
        """
        return Document(parser=self)
