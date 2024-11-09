# -*- coding: utf-8 -*-
# Create Date: 2024/10/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/log.py
# Description: 日志相关配置

from loguru import logger
import sys
from tqdm import tqdm


def set_logger(console: bool = False,
               file: bool = True,
               *,
               level: str = 'TRACE',
               use_tqdm: bool = True,
               file_path: str = None,
               format: str = None,
               zip: str = None):
    """ 设置日志

    Args:
        console (bool, optional): 输出到控制台. Defaults to False.
        file (bool, optional): 输出到文件. Defaults to True.
        level (str, optional): 日志等级. Defaults to 'TRACE'.
        use_tqdm (bool, optional): 使用 tqdm.write 作为输出 sink (不支持 rich), 更好跟进度条共存. Defaults to True.
        file_path (str, optional): 文件路径. Defaults to None.
        format (str, optional): 日志格式. Defaults to None.
        zip (str, optional): 文件压缩格式. Defaults to False.
    """
    logger.remove()

    if use_tqdm:
        out = lambda msg: tqdm.write(msg, end='')
    else:
        out = sys.stdout

    if format is None:
        format="{time:YYYY-MM-DD HH:mm:ss} | <lvl><normal>{level: <8}</normal></lvl> | {message}"

    if console:
        logger.add(out, format=format, level=level, colorize=True,)
    if file:
        if file_path is None:
            file_path = 'logs/{time}.log'
        logger.add(file_path, format=format, retention=10, compression=zip, level=level)
