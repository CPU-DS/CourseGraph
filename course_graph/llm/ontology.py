# -*- coding: utf-8 -*-
# Create Date: 2024/11/02
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/ontology.py
# Description: 知识图谱本体设计

from dataclasses import dataclass, field


@dataclass
class Ontology:
    entities: dict[str, str] = field(default_factory=lambda: {"知识点": "知识点实体类型表示特定领域或学科中的知识单元"})
    relations: dict[str, str] = field(default_factory=lambda: {
        "包含": "某一个知识点包含另一个知识点",
        "相关": "知识点之间存在相互联系、相互影响和相互作用",
        "顺序": "学习知识点具有明显的先后关系，也就是学习某一个知识点后才能学习另一个，存在前驱后继的关系"
    })
    attributes: dict[str, str] = field(default_factory=lambda: {"定义": "清楚的规定出知识点概念、意义的描述语句"})

ontology = Ontology()
