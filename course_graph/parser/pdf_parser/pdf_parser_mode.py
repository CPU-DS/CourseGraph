# -*- coding: utf-8 -*-
# Create Date: 2024/10/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/pdf_parser/pdf_parser_mode.py
# Description: PDF文档解析器模式接口

from abc import ABC
from ...llm import MLM, VLUPrompt, LLM


class PDFParserMode(ABC):
    """ PDF文档解析器模式基类
    """
    pass


class BaseMode(PDFParserMode):

    def __init__(self):
        """ 使用基础模式解析
        """
        pass


class PaddleMode(PDFParserMode):

    def __init__(self):
        """ 使用飞桨的版面分析解析
        """
        pass


class VisualMode(PDFParserMode):

    def __init__(self, model: MLM, prompt: VLUPrompt):
        """ 使用多模态大模型解析, 实现参考: https://github.com/lazyFrogLOL/llmdocparser

        Args:
            model (MLM): 多模态大模型
            prompt (VLUPrompt): 大模型对应的提示词
        """
        self.visual_model = model
        self.prompt = prompt.set_type_ocr()


class CombinationMode(PDFParserMode):
    """ 使用 OCR + 大模型综合解析 (推荐)

        Args:
            visual_model (MLM): 多模态大模型
            visual_prompt (VLUPrompt): 大模型对应的提示词
            llm (LLM | None, optional): 使用大模型矫正OCR结果. Defaults to None.
    """

    def __init__(self,
                 visual_model: MLM,
                 visual_prompt: VLUPrompt,
                 llm: LLM | None = None):
        self.visual_model = visual_model
        self.visual_prompt = visual_prompt.set_type_ocr()
        self.llm = llm
