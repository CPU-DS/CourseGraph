# -*- coding: utf-8 -*-
# Create Date: 2025/03/06
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/overview.py
# Description: 原始数据概览


from pathlib import Path
from glob import glob
import json
from tabulate import tabulate

data_path = 'experimental/data'


def overview():
    entity_details = {}
    relation_details = {}
    details = []
    for file in glob(data_path + '/*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            entity_lens = [len(d['entities']) for d in data]
            relation_lens = [len(d['relations']) for d in data]
            
            details.append({
                'file': Path(file).name.replace('.json', ''),
                'sum': len(data),
                'entity_sum': sum(entity_lens),
                'relation_sum': sum(relation_lens),
                'entity_avg': sum(entity_lens) / len(data),
                'relation_avg': sum(relation_lens) / len(data),
                'text_len_avg': sum([len(d['text']) for d in data]) / len(data)
            })
            
            for d in data:
                for n in d['entities']:
                    entity_details[n['type']] = entity_details.get(n['type'], 0) + 1
                for r in d['relations']:
                    relation_details[r['type']] = relation_details.get(r['type'], 0) + 1
    entity_details = [{
        'type': k,
        'sum': v
    } for k, v in entity_details.items()]
    relation_details = [{
        'type': k,
        'sum': v
    } for k, v in relation_details.items()]
    print(tabulate(details, headers='keys', tablefmt='grid'))
    print(f'总文本数量: {sum([l["sum"] for l in details])}')
    print(f'总实体数量: {sum([l["entity_sum"] for l in details])}')
    print(f'总关系数量: {sum([l["relation_sum"] for l in details])}')
    print(tabulate(entity_details, headers='keys', tablefmt='grid'))
    print(tabulate(relation_details, headers='keys', tablefmt='grid'))


if __name__ == '__main__':
    overview()
