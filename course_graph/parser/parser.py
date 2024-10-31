# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/parser.py
# Description: 定义文档解析器基类

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from .base import Document, BookMark
import os
import uuid


class ContentType(Enum):
    """ 内容类型
    """
    Text = 1
    Title = 2


@dataclass
class Content:
    """ 内容
    """
    type: ContentType  # 只有title两类
    origin_type: str  # 和 bbox 同为目标检测结果
    content: str  # 文本
    bbox: tuple[float]


@dataclass
class Page:
    """ 页面
    """
    page_index: int
    contents: list[Content]


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
    def get_content(self, bookmark: BookMark) -> list[Content]:
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
        return Document(id='0:' + str(uuid.uuid4()),
                        name=os.path.basename(self.file_path).split('.')[0],
                        file_path=self.file_path,
                        bookmarks=self.get_bookmarks(),
                        parser=self)
