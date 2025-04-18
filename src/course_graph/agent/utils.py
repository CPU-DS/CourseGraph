# -*- coding: utf-8 -*-
# Create Date: 2025/03/28
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/utils.py
# Description: 工具

import asyncio
import functools
from typing import Callable
from .trace import TraceEvent, TraceEventType
from loguru import logger


def async_to_sync(func: Callable) -> Callable:
    """ 将异步函数转换为同步函数
    
    Args:
        func (Callable): 异步函数

    Returns:
        Callable: 同步函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError
        return asyncio.run(func(*args, **kwargs))
    return wrapper


def trace_callback(trace: TraceEvent) -> None:
    """ 默认 Trace 回调
    """
    data = ""
    match trace.event_type:
        case TraceEventType.USER_MESSAGE | TraceEventType.AGENT_MESSAGE:
            data = trace.data['message']
        case TraceEventType.TOOL_CALL | TraceEventType.MCP_TOOL_CALL:
            args = ", ".join([f"{k}='{v}'" for k,v in trace.data['arguments'].items()])
            data = f"{trace.data['function']}({args})"
        case TraceEventType.TOOL_RESULT:
            result = trace.data['result']
            data = result['content']
            if result.get('context_variables'):
                data += f" \t| {result['context_variables']}"
            if result.get('message'):
                data += f" \t| message={result['message']}"
        case TraceEventType.CONTEXT_UPDATE:
            data = trace.data['context']
        case TraceEventType.AGENT_SWITCH:
            data = f"{trace.agent.name}({trace.agent.id}) -> {trace.data['to_agent'].name}({trace.data['to_agent'].id})"
        case _:
            pass
    logger.info(f"{trace.agent.name}({trace.agent.id})\t| {trace.event_type.name}\t| {data}")
