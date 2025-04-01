# -*- coding: utf-8 -*-
# Create Date: 2025/03/29
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/agent/use_mcp.py
# Description: 使用 MCP 工具

from course_graph.agent import Agent, Controller, MCPServer
from course_graph.llm import Qwen
import asyncio

qwen = Qwen()


async def main():
    async with MCPServer(
        {
            "command": "uv",
            "args": ["--directory", "examples/agent", "run", "mcp_server.py"],
        }
    ) as mcp_server:
        agent = Agent(llm=qwen, mcp_server=[mcp_server])
        controller = Controller()
        _, resp = await controller.run(agent, "帮我查询今天南京的天气")
        print(resp)


if __name__ == "__main__":
    asyncio.run(main())
