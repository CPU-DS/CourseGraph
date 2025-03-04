# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/get_knowledge_graph_pdf.py
# Description: 为pdf文档抽取知识点图谱

from course_graph.parser import PDFParser
from course_graph.database import Neo4j
from course_graph.llm import Qwen
from course_graph import set_logger
import argparse

set_logger(console=True, file=False)

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url', default='localhost:7687')
parser.add_argument('-n', '--user', default='neo4j')
parser.add_argument('-p', '--password', default='neo4j')
parser.add_argument('-f', '--file')

args = parser.parse_args()
assert args.file is not None and args.file.endswith('.pdf'), 'Please input a pdf file.'

neo4j = Neo4j(args.url, args.user, args.password)
model = Qwen()

with PDFParser(args.file) as parser:
    document = parser.get_document()
    document.set_knowledgepoints_by_llm(model)
    document.to_graph(neo4j)
