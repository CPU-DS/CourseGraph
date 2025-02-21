# -*- coding: utf-8 -*-
# Create Date: 2024/12/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/kg/api.py
# Description: 知识图谱查询接口

from fastapi import FastAPI, Body, Security, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from ..database import Neo4j
import uvicorn
from .model import *

API_KEYS = ['sk-1234567890']

api_key_header = APIKeyHeader(name='api-key', auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail='Invalid API key'
        )
    return api_key_header

class API:
    def __init__(self,
                 neo4j: Neo4j, 
                 host: str = '0.0.0.0', 
                 port: int = 8000, 
                 cors: bool = False
        ) -> None:
        self.app = FastAPI()
        if cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        self.neo4j = neo4j
        self.host = host
        self.port = port
        
        self.base_router = APIRouter(
            prefix='/api',
        )
        
        @self.base_router.post('/query/categories')
        async def query_categories(api_key: str = Security(get_api_key)) -> Response:
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query categories success',
                data=['课程', '章节', '知识点']
            )
        
        self.app.include_router(
            router=self.base_router
        )
        
    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)


