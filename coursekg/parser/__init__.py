# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/parser/__init__.py
# Description: 文档解析器接口

from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .base import BookMark, Document
from .parser import Page, Parser
from .utils import get_parser
from .config import Config
