# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/__init__.py
# Description: 文档解析器接口

from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .document import Document
from .types import BookMark
from .parser import Parser
from .types import Page
from .config import CONFIG
from .utils import instance_method_transactional
