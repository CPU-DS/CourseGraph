# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/config.py
# Description: 类封装

from modelscope import AutoModel, AutoTokenizer
from dataclasses import dataclass
from pymongo.collection import Collection
from ..database import Faiss


@dataclass
class Model:
    model: AutoModel
    tokenizer: AutoTokenizer


@dataclass
class Database:
    faiss: Faiss
    mongo: Collection
