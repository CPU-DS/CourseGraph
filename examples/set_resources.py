# -*- coding: utf-8 -*-
# Create Date: 2024/07/31
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/set_resources.py
# Description: 为实体挂载资源

from course_graph.parser import get_parser
from course_graph.resource import PPTX, ResourceMap
from course_graph.llm import VLLM, MLM, ExamplePrompt, MiniCPMPrompt

model = VLLM(path='model/Qwen/Qwen2-7B-Instruct')
visual_model = MLM(path='model/openbmb/MiniCPM-V-2_6')
visual_prompt = MiniCPMPrompt()

parser = get_parser('assets/探索数据的奥秘.docx')
document = parser.get_document()
document.set_knowledgepoints_by_llm(model, ExamplePrompt())
pptx = PPTX('assets/pptx/探索数据的奥秘.docx/Chpt6_5_决策树.pptx')
pptx.set_maps_by_visual_model(visual_model, visual_prompt)
document.set_resource(ResourceMap('6.5决策树', pptx))

model.close()
