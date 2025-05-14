# -*- coding: utf-8 -*-
# Create Date: 2025/03/26
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/utils.py
# Description: 提示词工具

import re
import json
from mistune import create_markdown
from typing import Optional
import ast as ast_f


def post_process(response: str) -> Optional[list | dict]:
    """ 将模型返回处理成列表或字典格式

    Args:
        response (str): 模型输出

    Returns:
        list | dict | None: 格式输出
    """
    replace_tuple = []  # 替换掉可能出现的非法字符
    md = create_markdown(renderer='ast')
    ast = md(response)
    json_fragments = []
    for node in ast:
        if node["type"] == "block_code" and node.get("info", "").strip() in ("json", ""):
            json_fragments.append(node["raw"])
    if not json_fragments:
        return None

    fragment = json_fragments[-1] # 可能会返回多个结果从语义上只取最后一个结果
    for a, b in replace_tuple:
        fragment = fragment.replace(a, b)
    try:
        return json.loads(fragment)
    except json.decoder.JSONDecodeError:
        return ast_f.literal_eval(fragment)


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
