# -*- coding: utf-8 -*-
# Create Date: 2024/12/20
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/kg/api.py
# Description: 知识图谱查询接口

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from ..database import Neo4j
import uvicorn
from .model import *

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
        
        @self.app.post('/api/server')
        async def server() -> Response[Server]:
            return Response(
                code=ResponseCode.SUCCESS,
                message='Get server information successfully.',
                data=Server(
                    serverUrl=self.neo4j.bolt_url,
                    serverUser=self.neo4j.username,
                    serverPassword=self.neo4j.password
                )
            )
        
        @self.app.post('/api/query/node')
        async def query_node(query: Query = Body(...)):
            node = self.neo4j.run(f'MATCH (n) where n.id = "{query.id}" RETURN n').data()[0]['n']
            resp = dict(node)
            del resp['color']
            resp['label'] = list(node.labels)[0]
            
            if query.id.startswith('2'):
                del resp['resource']
                resp['chapters'] = []
                resp['relations'] = []
                for ms in self.neo4j.run(f'MATCH (m:章节)-[]->(n) where n.id = "{query.id}" RETURN m, n').data():
                    chapter = dict(ms['m'])
                    resp['chapters'].append({
                        'id': chapter['id'],
                        'name': chapter['name'],
                    })
                    resp['chapters'].sort(key=lambda x: x['name'])
                for rs in self.neo4j.run(f'MATCH (n)-[r]->(a:知识点) where n.id = "{query.id}" RETURN r,a').data():
                    relation_type = type(rs['r']).__name__
                    relation_value = {'id': rs['a']['id'], 'name': rs['a']['name']}

                    for relation in resp['relations']:
                        if relation_type in relation:
                            relation[relation_type].append(relation_value)
                            break
                    else:
                        resp['relations'].append({relation_type: [relation_value]})
            elif query.id.startswith('1'):
                del resp['resource']
                resp['parents'] = []
                resp['children'] = []
                resp['knowledgepoints'] = []
                for ms in self.neo4j.run(f'MATCH (n)-[]->(m:章节) where n.id = "{query.id}" RETURN m').data():
                    children = dict(ms['m'])
                    resp['children'].append({
                        'id': children['id'],
                        'name': children['name'],
                    })
                    resp['children'].sort(key=lambda x: x['name'])
                for ms in self.neo4j.run(f'MATCH (m:章节)-[]->(n) where n.id = "{query.id}" RETURN m').data():
                    parent = dict(ms['m'])
                    resp['parents'].append({
                        'id': parent['id'],
                        'name': parent['name'],
                    })
                for ms in self.neo4j.run(f'MATCH (n)-[]->(m:知识点) where n.id = "{query.id}" RETURN m').data():
                    knowledgepoint = dict(ms['m'])
                    resp['knowledgepoints'].append({
                        'id': knowledgepoint['id'],
                        'name': knowledgepoint['name'],
                    })
                    resp['knowledgepoints'].sort(key=lambda x: x['name'])
            elif query.id.startswith('0'):
                resp['chapters'] = []
                for ms in self.neo4j.run(f'MATCH (n)-[]->(m:章节) where n.id = "{query.id}" RETURN m').data():
                    chapter = dict(ms['m'])
                    resp['chapters'].append({
                        'id': chapter['id'],
                        'name': chapter['name'],
                    })
                    resp['chapters'].sort(key=lambda x: x['name'])
            return resp
        
        @self.app.post('/api/query/edge')
        async def query_edge(id: str = Body(...)):
            pass
        
    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)


