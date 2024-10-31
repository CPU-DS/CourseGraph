# -*- coding: utf-8 -*-
# Create Date: 2024/08/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/pdf_knowledgepoint.py
# Description: 使用图文理解模型辅助解析pdf

from course_graph.parser.pdf_parser import PDFParser, VisualMode
from course_graph.llm import VLM

vlm = VLM(path='model/openbmb/MiniCPM-V-2_6')
parser = PDFParser('assets/深度学习入门：基于Python的理论与实现.pdf', vlm=vlm)
parser.close()
