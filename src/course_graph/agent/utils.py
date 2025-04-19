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
