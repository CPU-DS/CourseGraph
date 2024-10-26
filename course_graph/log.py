# -*- coding: utf-8 -*-
# Create Date: 2024/10/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/log.py
# Description: 日志相关配置

from loguru import logger
import sys
import time


def setup_logger(console: bool = False,
                 file: bool = True,
                 file_path: str = None):
    """ 设置日志

    Args:
        console (bool, optional): 输出到控制台. Defaults to False.
        file (bool, optional): 输出到文件. Defaults to True.
        file_path (str, optional): 文件路径. Defaults to None.
    """
    logger.remove()

    if console:
        logger.add(sys.stdout,
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    if file:
        if file_path is None:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file_path = f'logs/run_{current_time}.log'
        logger.add(file_path,
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
