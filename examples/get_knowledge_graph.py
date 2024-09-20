# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/pdf_knowledgepoint.py
# Description: 为pdf文档抽取知识点图谱

from coursekg.parser import get_parser
from coursekg.database import Neo4j
from coursekg.llm import VLLM

model = VLLM('model/Qwen/Qwen2-7B-Instruct')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = [
    'assets/深度学习入门：基于Python的理论与实现.pdf', 'assets/机器学习.pdf',
    'assets/探索数据的奥秘.docx'
]

for file in files:
    with get_parser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(model)
        neo.run(document.to_cyphers())

model.close()
