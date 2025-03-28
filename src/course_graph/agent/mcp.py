# -*- coding: utf-8 -*-
# Create Date: 2025/03/28
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/mcp.py
# Description: MCP Server 和 Client 实现相关

from typing import Literal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from mcp.types import Tool

class MCPServer:
    def __init__(self, type: Literal['stdio'], command: str, args: list[str], envs: dict[str, str] = None):
        """ MCP 服务器

        Args:
            type (Literal['stdio']): 服务器类型, 目前只支持 'stdio'
            command (str): 命令
            args (list[str]): 参数
            envs (dict[str, str], optional): 环境变量. Defaults to None.
        """
        self.params = StdioServerParameters(command=command, args=args, envs=envs)
        self.stack = AsyncExitStack()
        self.session = None
        
    async def __aenter__(self):
        stdio_transport = await self.stack.enter_async_context(stdio_client(self.params))
        self.stdio, self.write = stdio_transport
        self.session = await self.stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        return self
    
    async def list_tools(self) -> list[Tool]:
        return (await self.session.list_tools()).tools
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.stack.aclose()
        
        
        
        
        
