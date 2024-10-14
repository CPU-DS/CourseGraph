# -*- coding: utf-8 -*-
# Create Date: 2024/08/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/pdf_knowledgepoint.py
# Description: 使用多模态大模型解析pdf

from course_graph.parser.pdf_parser import PDFParser, VisualMode
from course_graph.llm import MiniCPM, MiniCPMPrompt

file = 'assets/深度学习入门：基于Python的理论与实现.pdf'
visual_model = MiniCPM()
parser = PDFParser(file)
parser.parser_mode = VisualMode(visual_model, MiniCPMPrompt())
parser.close()
