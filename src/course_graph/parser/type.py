# -*- coding: utf-8 -*-
# Create Date: 2024/11/04
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/type.py
# Description: 定义各种中间类型

from dataclasses import dataclass
from .entity import KPEntity
from ..resource import Resource
from enum import Enum
from dataclasses import dataclass


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


@dataclass
class PageIndex:
    index: int
    anchor: tuple[float, float]  # 等价于 bbox 中的 x1,y1


@dataclass
class BookMark:
    """ 书签
    """
    id: str
    title: str
    page_start: PageIndex
    page_end: PageIndex
    level: int
    subs: list['BookMark'] | list[KPEntity]
    resource: list[Resource]

    def set_page_end(self, page_end: PageIndex) -> None:
        """ 设置书签的结束页, 和直接修改 BookMark 对象的 page_end 属性不同, 该方法会考虑到书签嵌套的情况

        Args:
            page_end (int): 结束页
        """
        self.page_end = page_end
        if self.subs and isinstance(self.subs[-1], BookMark):
            self.subs[-1].set_page_end(page_end)

    def get_kps(self) -> list[KPEntity]:
        """ 获取当前书签下的所有 知识点实体

        Returns:
            list[KPEntity]: 实体列表
        """
        kps: list[KPEntity] = []

        def get_kp(bookmark: BookMark):
            for sub in bookmark.subs:
                match sub:
                    case KPEntity():
                        kps.append(sub)
                    case BookMark():
                        get_kp(sub)
        get_kp(self)
        return kps

    def __repr__(self, detail: bool = True) -> str:
        if detail:
            s = ', \n\t\t'.join([sub.__repr__(False) for sub in self.subs])
            return f'''BookMark(title="{self.title}", 
\t page_start={self.page_start.index}, 
\t page_end={self.page_end.index}, 
\t level={self.level}, 
\t resource={self.resource}),
\t subs=[\t{s}]'''
        else:
            return  f'BookMark(title="{self.title}", ...)'
