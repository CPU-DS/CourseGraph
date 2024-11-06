# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/config.py
# Description: 定义知识图谱抽取配置

from dataclasses import dataclass, field


@dataclass
class Config:
    ignore_page: list[str] = field(default_factory=lambda: [
        '封面', 
        '封面页', 
        '目录', 
        '目录页', 
        '扉页', 
        '版权声明', 
        '版权页', 
        '数字版权声明', 
        '前言', 
        '序言',
        '译者序', 
        '作译者介绍', 
        'O’Reilly Media, Inc.介绍', 
        '索引', 
        '后记', 
        '思考题',
        '参考文献'
    ])

config = Config()