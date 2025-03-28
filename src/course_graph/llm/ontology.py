# -*- coding: utf-8 -*-
# Create Date: 2024/11/02
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/ontology.py
# Description: 知识图谱本体设计

ONTOLOGY = {  # 针对知识点层级
    'entities': {
        '知识点': '特定领域或学科中的知识单元，需要具有一定的深度与广度，是一个相对自包含的完整结构'
    },
    'relations': {
        '包含': '某一个知识点包含另一个知识点',
        '相关': '知识点之间存在相互联系、相互影响和相互作用',
        '顺序': '学习知识点具有明显的先后关系，也就是学习某一个知识点后才能学习另一个，存在前驱后继的关系'
    },
    'attributes': {
        '定义': '清楚的规定出知识点概念、意义的描述语句'
    }
}