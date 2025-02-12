# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/pdf_knowledgepoint.py
# Description: 使用动态提示词示例抽取知识点图谱

from course_graph.parser import PDFParser
from course_graph.database import Neo4j
from course_graph.llm import VLLM
from course_graph.llm.prompt import ExamplePromptGenerator, SentenceEmbeddingStrategy

model = VLLM(path='model/Qwen/Qwen2-7B-Instruct')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
strategy = SentenceEmbeddingStrategy(
    embed_model_path='model/lier007/xiaobu-embedding-v2')

with PDFParser('assets/深度学习入门：基于Python的理论与实现.pdf') as parser:
    document = parser.get_document()
    example_prompt = ExamplePromptGenerator(strategy)
    document.set_knowledgepoints_by_llm(model,
                                        example_prompt,
                                        self_consistency=True,
                                        samples=6,
                                        top=0.8)
    neo.run(document.to_cyphers())

model.close()
