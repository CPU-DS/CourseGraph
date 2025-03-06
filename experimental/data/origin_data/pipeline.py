# -*- coding: utf-8 -*-
# Create Date: 2025/03/06
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/data/origin_data/pipeline.py
# Description: 原始数据处理


import json
from glob import glob
from pathlib import Path

data_path = Path(__file__).parent


def serialize(file: str):
    texts = []
    with open(file, 'r') as f:
        data = json.load(f)
        for line in data:
            if len(result := line['annotations'][0]['result']) != 0:
                nodes = []
                relations = []
                id2name = {}
                for r in result:
                    if r['type'] == 'labels':
                        nodes.append({
                            'start': r['value']['start'],
                            'end': r['value']['end'],
                            'text': r['value']['text'],
                            'type': r['value']['labels'][0]})
                        id2name[r['id']] = r['value']['text']
                    elif r['type'] == 'relation':
                        type_ = r.get('labels', [None])[0]
                        if type_:
                            relations.append({
                                'source': id2name[r['from_id']],
                                'target': id2name[r['to_id']],
                                'type': type_})
                texts.append({
                    'text': line['data']['text'],
                    'nodes': nodes,
                    'relations': relations})
    return texts


def pipeline():
    for file in glob(str(data_path / '*.json')):
        base_name = Path(file).name
        texts = serialize(file)
        with open(str(data_path.parent / base_name), 'w', encoding='utf-8') as f:
            json.dump(texts, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    pipeline()
