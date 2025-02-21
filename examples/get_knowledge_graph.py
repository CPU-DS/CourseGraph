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
parser.add_argument('-u', '--url', default='bolt://localhost:7687')
parser.add_argument('-n', '--user', default='neo4j')
parser.add_argument('-p', '--password', default='neo4j')
parser.add_argument('-f', '--file')

args = parser.parse_args()

set_logger(console=True, file=False)

model = Qwen()
neo = Neo4j(args.url, args.user, args.password)

with PDFParser(args.file) as parser:
    document = parser.get_document()
    document.set_knowledgepoints_by_llm(model)
    neo.run(document.to_cyphers())

model.close()
