# -*- coding: utf-8 -*-
# Create Date: 2024/12/30 Good bye 2024!
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

class Server(BaseModel):
    serverUrl: str
    serverUser: str
    serverPassword: str
    
class Query(BaseModel):
    id: str
