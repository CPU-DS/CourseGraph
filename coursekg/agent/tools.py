# -*- coding: utf-8 -*-
# Create Date: 2024/09/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/agent/tools.py
# Description: 提供常用的工具函数
import requests
import json
from typing import TypedDict, Callable
from openai.types.chat import ChatCompletionToolParam


class Tool(TypedDict):
    tool: ChatCompletionToolParam
    function: Callable


def baidu_baike_api(keyword: str) -> str:
    res = requests.get(
        f'https://baike.baidu.com/api/openapi/BaikeLemmaCardApi?scope=103&format=json&appid=379020&bk_key={keyword}&bk_length=1000'
    ).text
    if len(res) == 0:
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
