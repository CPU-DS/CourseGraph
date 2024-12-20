# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/pdf_knowledgepoint.py
# Description: 为pdf文档抽取知识点图谱

from course_graph.parser import PDFParser
from course_graph.database import Neo4j
from course_graph.llm import Qwen
from course_graph import set_logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url', default='http://localhost:7474', help='Neo4j server URL')
parser.add_argument('-n', '--user', default='neo4j', help='Neo4j username')
parser.add_argument('-p', '--password', default='neo4j', help='Neo4j password')
args = parser.parse_args()

set_logger(console=True, file=False)

model = Qwen()
neo = Neo4j(args.url, args.user, args.password)

with PDFParser('assets/深度学习入门：基于Python的理论与实现.pdf') as parser:
    document = parser.get_document()
    document.set_knowledgepoints_by_llm(model)
    neo.run(document.to_cyphers())

model.close()
