# -*- coding: utf-8 -*-
# Create Date: 2025/03/26
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/utils.py
# Description: 提示词工具

import re
import json


def post_process(response: str) -> list | dict | None:
    """ 将模型返回处理成列表或字典格式

    Args:
        response (str): 模型输出

    Returns:
        list | dict | None: 格式输出
    """
    replace_tuple = [('\\', ''), ('“', '"'), ('”', '"')]  # 替换掉可能出现的非法字符
    fragments = re.findall(r'```.*?\n([\s\S]*?)\n?```', response)
    if len(fragments) > 0:
        fragment: str = fragments[-1]  # 可能会返回多个结果从语义上只取最后一个结果
        for a, b in replace_tuple:
            fragment = fragment.replace(a, b)
        try:
            res = json.loads(fragment)
            return res
        except json.decoder.JSONDecodeError:
            return None


def json2md(data: dict | list) -> str:
    """ 将json格式转换为md格式

    Args:
        data (dict | list): 字典或列表
        
    Returns:
        str: 转换后的markdown格式文本
    """

    md_text = ""

    def process_item(item, level=0):
        nonlocal md_text

        if isinstance(item, dict):
            for key, value in item.items():
                md_text += "#" * (level + 1) + f" {key}\n\n"

                if isinstance(value, (dict, list)):
                    process_item(value, level + 1)
                else:
                    md_text += f"{value}\n\n"

        elif isinstance(item, list):
            for value in item:
                if isinstance(value, (dict, list)):
                    process_item(value, level)
                else:
                    md_text += f"- {value}\n"
            md_text += "\n"

    process_item(data)
    return md_text.strip()
