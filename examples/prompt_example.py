# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/pdf_knowledgepoint.py
# Description: 使用动态提示词示例抽取知识点图谱

from coursekg.document_parser import PDFParser
from coursekg.database import Neo4j
from coursekg.llm import ExamplePrompt, VLLM, SentenceEmbeddingStrategy

model = VLLM('model/Qwen/Qwen2-7B-Instruct')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
strategy = SentenceEmbeddingStrategy(
    embed_model_path='model/lier007/xiaobu-embedding-v2')

with PDFParser('assets/深度学习入门：基于Python的理论与实现.pdf') as parser:
    document = parser.get_document()
    document.set_knowledgepoints_by_llm(model,
                                        ExamplePrompt(strategy),
                                        self_consistency=True,
                                        samples=6,
                                        top=0.8)
    neo.run(document.to_cyphers())
