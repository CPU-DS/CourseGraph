# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/prompt_example_reimport.py
# Description: 导入提示词示例库

from course_graph.llm import SentenceEmbeddingStrategy

strategy = SentenceEmbeddingStrategy(
    embed_model_path='model/lier007/xiaobu-embedding-v2')
strategy.reimport_example(1792)
