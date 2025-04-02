# -*- coding: utf-8 -*-
# Create Date: 2025/03/25
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/knowledge_server.py
# Description: 知识图谱后端服务

from course_graph.kg import API, APIKeyManager
from course_graph.database import Neo4j
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url', default='bolt://localhost:7687')
parser.add_argument('-n', '--user', default='neo4j')
parser.add_argument('-p', '--password', default='neo4j')
args = parser.parse_args()

with Neo4j(args.url, args.user, args.password) as neo4j:
    api_key_manager = APIKeyManager()
    print(api_key_manager.generate_key())
    api = API(neo4j=neo4j, port=8000, api_key_manager=api_key_manager, cors=True)
    api.run()
