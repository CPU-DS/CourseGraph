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
from concurrent.futures import ThreadPoolExecutor
import asyncio

api_key_header = APIKeyHeader(name='api-key', auto_error=False)


class API:
    def __init__(self,
                 neo4j: Neo4j,
                 api_key_manager: APIKeyManager = None,
                 host: str = '0.0.0.0',
                 port: int = 8000,
                 cors: bool = False,
                 pool_max_workers: int = 10
        ) -> None:
        """ 初始化 API 服务
        
        Args:
            neo4j (Neo4j): Neo4j 数据库连接
            api_key_manager (APIKeyManager): API Key 管理器. Defaults to None.
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
        self.api_key_manager = api_key_manager
        self.executor = ThreadPoolExecutor(max_workers=pool_max_workers)

        self.base_router = APIRouter(
            prefix='/api'
        )
        self.echarts_router = APIRouter(
            prefix='/echarts',
            tags=['echarts']
        )

        self.__init_echarts_router()
        self.base_router.include_router(self.echarts_router)
        self.app.include_router(
            router=self.base_router
        )
    
    def __get_api_key(self, api_key: str = Security(api_key_header)):
        if self.api_key_manager and not self.api_key_manager.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Invalid API key'
            )
        return api_key
    
    def __init_echarts_router(self):
        @self.echarts_router.post('/categories')
        async def echarts_categories(_=Security(self.__get_api_key)) -> Response:
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query categories success',
                data=[{'name': '文档'}, {'name': '章节'}, {'name': '知识点'}]
            )

        @self.echarts_router.post('/nodes')
        async def echarts_nodes(page: Page, _=Security(self.__get_api_key)) -> Response:
            nodes = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.neo4j.get_nodes_with_relation_count,
                page.page_index * page.page_size,
                page.page_size
            )
            label2idx = {
                '文档': 0,
                '章节': 1,
                '知识点': 2
            }
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query nodes success',
                data=[{
                    'category': label2idx[list(node.labels)[0]],
                    **node._properties,
                    'relation_count': relation_count
                } for node, relation_count in nodes]
            )
        
        @self.echarts_router.post('/relations')
        async def echarts_relations(page: Page, _=Security(self.__get_api_key)) -> Response:
            relations = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.neo4j.get_relations,
                page.page_index * page.page_size,
                page.page_size
            )
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query relations success',
                data=[{
                    'source': relation.start_node._properties['id'],
                    'source_node': relation.start_node._properties,
                    'target': relation.end_node._properties['id'],
                    'target_node': relation.end_node._properties,
                    'relation_type': relation.type
                } for relation in relations]
            )
            
        @self.echarts_router.post('/nodes_count')
        async def echarts_nodes_count(_=Security(self.__get_api_key)) -> Response:
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query nodes count success',
                data=self.neo4j.get_nodes_count()
            )
        
        @self.echarts_router.post('/relations_count')
        async def echarts_relations_count(_=Security(self.__get_api_key)) -> Response:
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query relations count success',
                data=self.neo4j.get_relations_count()
            )
            
        @self.echarts_router.post('/max_relation_count')
        async def echarts_max_relation_count(_=Security(self.__get_api_key)) -> Response:
            return Response(
                code=ResponseCode.SUCCESS,
                message='Query max relation count success',
                data=self.neo4j.get_max_relation_count()
            )

    def run(self):
        """ 启动 API 服务 """
        uvicorn.run(self.app, host=self.host, port=self.port)