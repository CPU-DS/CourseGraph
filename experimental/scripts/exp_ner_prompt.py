# -*- coding: utf-8 -*-
# Create Date: 2025/05/12
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/exp_ner_prompt.ipynb
# Description: 命名实体识别


from prompt.ner import main
from course_graph.llm import Gemini, LLM
from glob import glob
import json
from course_graph.llm.prompt import (
    SentenceEmbeddingStrategy,
    F1Filter,
    ExamplePrompt
)
import os


def load_data(path: str) -> list[dict]:
    data= [
        line for file in glob(path + '/*.json') for line in json.load(open(file, 'r'))
    ]
    for item in data:
        for e in item['entities']:
            e['type'] = '知识点'
    return data

# gemini = Gemini()
# gemini.model = 'gemini-2.5-flash-preview-04-17'
# gemini.config = {
#     'reasoning_effort': 'high',
#     'json': True
# }

text_embed = Gemini(proxy='http://127.0.0.1:7890/')
text_embed.model = 'text-embedding-004'

strategy = SentenceEmbeddingStrategy(
    embed_model=text_embed,
    embed_dim=768,
    avoid_first=True,
    topk=3
)
strategy.json_block = True


api_key=os.getenv('AZURE_API_KEY')
base_url=os.getenv('AZURE_ENDPOINT')
deepseek_azure = LLM(
    base_url=base_url,
    api_key=api_key
)
deepseek_azure.model = 'DeepSeek-R1'
deepseek_azure.config['reasoning_parser'] = 'deepseek_r1'

main(
    eval_data=load_data('experimental/data'),
    embed_model=text_embed,
    chat_model=deepseek_azure,
    prompt_strategy=strategy
)

# def f1_func(example: dict, resp: str) -> float:
#     label = [(e['text'], e['type']) for e in example['entities']]
#     resp = json.loads(resp)
#     pred = []
#     for k, v in resp.items():
#         for n in v:
#             pred.append((n, k))
#     eval_result = metric(pred, label)
#     return eval_result['f1']

# filter = F1Filter(
#     llm=gemini,
#     prompt=ExamplePrompt(),
#     f1_func=f1_func,
#     filter_strategy='percentage',
#     filter_percent=0.4
# )

# origin_data = load_data(DATA_DIR)

# data = filter.calculate_f1(origin_data)
# with open(os.path.join(DATA_DIR, 'f1/data.json'), 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False)