# -*- coding: utf-8 -*-
# Create Date: 2025/03/06
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/overview.py
# Description: 原始数据概览


import sys
print(sys.executable)

from pathlib import Path
from glob import glob
import json
from tabulate import tabulate

data_path = Path(__file__).parent.parent / 'data'


def overview():
    node_details = {}
    relation_details = {}
    details = []
    for file in glob(str(data_path / '*.json')):
        with open(file, 'r') as f:
            data = json.load(f)
            nodes_lens = [len(d['nodes']) for d in data]
            relation_lens = [len(d['relations']) for d in data]
            
            details.append({
                'file': Path(file).name.replace('.json', ''),
                'sum': len(data),
                'node_sum': sum(nodes_lens),
                'relation_sum': sum(relation_lens),
                'node_avg': sum(nodes_lens) / len(data),
                'relation_avg': sum(relation_lens) / len(data),
                'text_len_avg': sum([len(d['text']) for d in data]) / len(data)
            })
            
            for d in data:
                for n in d['nodes']:
                    node_details[n['type']] = node_details.get(n['type'], 0) + 1
                for r in d['relations']:
                    relation_details[r['type']] = relation_details.get(r['type'], 0) + 1
    node_details = [{
        'type': k,
        'sum': v
    } for k, v in node_details.items()]
    relation_details = [{
        'type': k,
        'sum': v
    } for k, v in relation_details.items()]
    print(tabulate(details, headers='keys', tablefmt='grid'))
    print(f'总文本数量: {sum([l["sum"] for l in details])}')
    print(f'总节点数量: {sum([l["node_sum"] for l in details])}')
    print(f'总关系数量: {sum([l["relation_sum"] for l in details])}')
    print(tabulate(node_details, headers='keys', tablefmt='grid'))
    print(tabulate(relation_details, headers='keys', tablefmt='grid'))


if __name__ == '__main__':
    overview()
