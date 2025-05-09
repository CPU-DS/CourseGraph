# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/__init__.py
# Description: course_graph

from .log import set_logger, logger
from .version import __version__
import os

def use_proxy(proxy_url: str = 'http://127.0.0.1:7890'):
    """ 使用代理
    """
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
