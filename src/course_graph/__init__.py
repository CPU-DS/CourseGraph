# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/__init__.py
# Description: course_graph

from .log import set_logger, logger
from .version import __version__
import os

    
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
EXPERIMENTAL_DIR = os.path.join(ROOT_DIR, 'experimental')
DATA_DIR = os.path.join(EXPERIMENTAL_DIR, 'data')
MILVUS_PATH = os.path.join(ROOT_DIR, 'src/course_graph/database')
