# -*- coding: utf-8 -*-
# Create Date: 2024/11/04
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/entity.py
# Description: 定义知识点实体和关系类

from dataclasses import dataclass, field
from ..resource import Slice


@dataclass
class KPEntity:
    id: str
    name: str
    type: str
    relations: list['KPRelation'] = field(default_factory=list)
    attributes: dict[str, list] = field(default_factory=dict)  # 同一个属性可能会存在多个属性值, 后续选择一个最好的值
    best_attributes: dict[str, str] = field(default=dict)
    resourceSlices: list[Slice] = field(default_factory=list)


@dataclass
class KPRelation:
    id: str
    type: str
    tail: KPEntity