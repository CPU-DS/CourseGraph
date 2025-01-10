# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/config.py
# Description: 定义知识图谱抽取配置

from typing import TypedDict

class ConfigType(TypedDict):
    IGNORE_PAGE: list[str]

CONFIG: ConfigType = {
    'IGNORE_PAGE': []
}