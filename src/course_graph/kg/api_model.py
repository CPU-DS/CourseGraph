# -*- coding: utf-8 -*-
# Create Date: 2024/12/30 Goodbye 2024!
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/kg/model.py
# Description: API 模型

from pydantic import BaseModel
from typing import TypeVar, Generic
from enum import Enum
T = TypeVar('T')

class ResponseCode(Enum):
    SUCCESS = 200

class Response(BaseModel, Generic[T]):
    code: ResponseCode
    message: str
    data: T
    
class Page(BaseModel):
    page_index: int
    page_size: int
