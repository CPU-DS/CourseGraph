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
    best_attributes: dict[str, str] = field(default_factory=dict)
    resourceSlices: list[Slice] = field(default_factory=list)

    def __repr__(self, detail: bool = True) -> str:
        if detail:
            s = ', \n\t\t'.join([relation.__repr__() for relation in self.relations])
            return f'''KPEntity(name="{self.name}", 
\t type="{self.type}", 
\t attributes={self.attributes}, 
\t best_attributes={self.best_attributes})
\t relations=[{s}]'''
        else:
            return f'KPEntity(name="{self.name}", ...)'


@dataclass
class KPRelation:
    id: str
    type: str
    tail: KPEntity

    def __repr__(self) -> str:
        return f'''KPRelation(type="{self.type}", tail={self.tail.__repr__(False)})'''
