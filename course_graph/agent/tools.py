# -*- coding: utf-8 -*-
# Create Date: 2024/09/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/tools.py
# Description: 提供常用的工具函数

import requests
import json
from typing import TypedDict, Callable
from openai.types.chat import ChatCompletionToolParam
from typing_extensions import Required


class Tool(TypedDict, total=False):
    tool: Required[ChatCompletionToolParam]
    function: Required[Callable]
    function_name: str  # 需要与tool.function.name相同，作为function的索引


def baidu_baike_api(keyword: str) -> str:
    res = requests.get(
        f'https://baike.baidu.com/api/openapi/BaikeLemmaCardApi?appid=379020&bk_key={keyword}'
    ).text
    if len(res) == '{}':  # json 空对象
        return '抱歉，暂时未查询到相关信息'
    return json.loads(res)['abstract']


BAIDU_BAIKE: Tool = {
    'tool': {
        'type': 'function',
        'function': {
            'name': 'baidu_baike_api',
            'description': '百度百科是百度公司推出的一部内容开放、自由的网络百科全书。可以根据用户给定的关键词查询相关信息。',
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "查询关键字"
                    }
                }
            }
        }
    },
    'function': baidu_baike_api
}
