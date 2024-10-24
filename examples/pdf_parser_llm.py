# -*- coding: utf-8 -*-
# Create Date: 2024/08/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/pdf_knowledgepoint.py
# Description: 使用多模态大模型解析pdf

from course_graph.parser.pdf_parser import PDFParser, VisualMode
from course_graph.llm import MLM, MiniCPMPrompt

visual_model = MLM(path='model/openbmb/MiniCPM-V-2_6')
visual_prompt = MiniCPMPrompt()
parser_mode = VisualMode(visual_model, visual_prompt)
parser = PDFParser('assets/深度学习入门：基于Python的理论与实现.pdf', parser_mode=parser_mode)
parser.close()
