# -*- coding: utf-8 -*-
# Create Date: 2024/08/05
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/utils.py
# Description: 工具函数

from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .docx_parser import DOCXParser
    from .pdf_parser import PDFParser


def get_parser(file_path: str) -> Union[DOCXParser, PDFParser]:
    """ 根据文档名称自动获取文档解析器

    Args:
        file_path (str): 文档路径

    Raises:
        ValueError: 文件格式不符合

    Returns:
        Parser: 文档解析器
    """
    if file_path.endswith('.docx'):
        return DOCXParser(file_path)
    elif file_path.endswith('.pdf'):
        return PDFParser(file_path)
    else:
        raise ValueError('只支持 .docx 和 .pdf 格式')
