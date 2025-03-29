# -*- coding: utf-8 -*-
# Create Date: 2025/03/28
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/mcp.py
# Description: MCP Server 和 Client 实现相关

from typing import TypedDict, Required, NotRequired
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack
from mcp.types import Tool


class STDIO(TypedDict):
    command: Required[str]
    args: Required[list[str]]
    envs: NotRequired[dict[str, str]]


class SSE(TypedDict):
    url: Required[str]
    headers: NotRequired[dict[str, str]]


class MCPServer:
    def __init__(self, server: STDIO | SSE):
        """ MCP 服务器

        Args:
            server (STDIO | SSE): MCP 服务器配置
        """
        self.server = server
        self.stack = AsyncExitStack()
        self.session: ClientSession = None
        self.tools: list[Tool] = None
        
    async def __aenter__(self):
        if 'command' in self.server.keys():
            self.params = StdioServerParameters(command=self.server['command'], args=self.server['args'], envs=self.server.get('envs'))
            transport = await self.stack.enter_async_context(stdio_client(self.params))
        else:
            transport = await self.stack.enter_async_context(sse_client(url=self.server['url'], headers=self.server.get('headers')))
        self.session = await self.stack.enter_async_context(ClientSession(transport[0], transport[1]))
        await self.session.initialize()
        
        self.tools = (await self.session.list_tools()).tools
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.stack.aclose()
