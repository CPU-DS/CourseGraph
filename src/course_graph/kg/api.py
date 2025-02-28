# -*- coding: utf-8 -*-
# Create Date: 2024/12/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/kg/api.py
# Description: 知识图谱查询接口

from fastapi import FastAPI, Security, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from ..database import Neo4j
import uvicorn
from .api_model import *
from .api_key import APIKeyManager

api_key_header = APIKeyHeader(name='api-key', auto_error=False)


class API:
    def __init__(self,
                 neo4j: Neo4j,
                 api_key_manager: APIKeyManager,
                 host: str = '0.0.0.0',
                 port: int = 8000,
                 cors: bool = False
                 ) -> None:
        """ 初始化 API 服务
        
        Args:
            neo4j (Neo4j): Neo4j 数据库连接
            api_key_manager (APIKeyManager): API Key 管理器
            host (str, optional): API 服务地址. Defaults to '0.0.0.0'.
            port (int, optional): API 服务端口. Defaults to 8000.
            cors (bool, optional): 是否启用跨域. Defaults to False.
        """
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

        def get_api_key(api_key: str = Security(api_key_header)):
            if not api_key_manager.validate_key(api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail='Invalid API key'
                )
            return api_key

        self.base_router = APIRouter(
            prefix='/api',
        )

        @self.base_router.post('/query/categories')
        async def query_categories(_: str = Security(get_api_key)) -> Response:
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query categories success',
                data=['课程', '章节', '知识点']
            )

        @self.base_router.post('/query/nodes')
        async def query_nodes(_: str = Security(get_api_key)) -> Response:
            pass

        self.app.include_router(
            router=self.base_router
        )

    def run(self):
        """ 启动 API 服务
        """
        uvicorn.run(self.app, host=self.host, port=self.port)
