# -*- coding: utf-8 -*-
# Create Date: 2025/03/29
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/agent/mcp_server.py
# Description: 一个简单的 MCP Server 示例

from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP('weather')


@mcp.tool()
def get_weather(city: str) -> str:
    """ 获取指定城市当天的天气
    
    Args:
        city: 城市名称

    Returns:
        dict: 天气信息
    """
    resp = {
        'city': city,
        'temperature_high': 20,
        'temperature_low': 18,
        'temperature_unit': 'C',
        'weather': 'sunny'
    }
    return json.dumps(resp, ensure_ascii=False)


if __name__ == '__main__':
    mcp.run(transport='stdio')
