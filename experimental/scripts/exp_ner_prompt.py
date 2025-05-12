# -*- coding: utf-8 -*-
# Create Date: 2025/05/12
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/exp_ner_prompt.py
# Description: 命名实体识别


from prompt.ner import main, metric
from course_graph import use_proxy
from course_graph.llm import Gemini
from glob import glob
import json
from course_graph.llm.prompt import (
    SentenceEmbeddingStrategy,
    F1Filter,
    ExamplePrompt
)


def load_data(path: str) -> list[dict]:
    return [
        line for file in glob(path + '/*.json') for line in json.load(open(file, 'r'))
    ]


use_proxy()

llm = Gemini()
llm.model = 'gemini-2.5-flash-preview-04-17'
llm.config = {
    'reasoning_effort': 'high',
    'json': True
}

text_embed = Gemini()
text_embed.model = 'text-embedding-004'

strategy = SentenceEmbeddingStrategy(
    milvus_path='src/course_graph/database/milvus_f1_filter.db',
    embed_model=text_embed,
    embed_dim=768,
    avoid_first=True,
    topk=3,
)
strategy.json_block = False

data_path = 'experimental/data'
data = load_data(data_path)

for item in data:
    for e in item['entities']:
        e['type'] = '知识点'
      
def f1_func(example: dict, resp: str) -> float:
    label = [(e['text'], e['type']) for e in item['entities']]
    resp = json.loads(resp)
    pred = []
    for k, v in resp.items():
        for n in v:
            pred.append((n, k))
    eval_result = metric(pred, label)
    return eval_result['f1']

filter = F1Filter(
    llm=llm,
    prompt=ExamplePrompt(),
    f1_func=f1_func,
    filter_strategy='percentage',
    filter_percent=0.4
)

main(
    eval_data=data,
    embed_model=text_embed,
    chat_model=llm,
    prompt_strategy=strategy,
    example_filter=filter,
    reimport_example=True
)
