# -*- coding: utf-8 -*-
# Create Date: 2025/04/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/main.py
# Description: 提示词测试


from course_graph.llm.prompt import (
    ExamplePrompt,
    post_process,
    SentenceEmbeddingStrategy
)
from course_graph.llm import Gemini
from glob import glob
from pprint import pprint
import json
import os


def load_data(data_path: str) -> list[dict]:
    data = []
    for file in glob(data_path + '/*.json'):
        with open(file, 'r') as f:
            data.extend(json.load(f))
    return data
    

if __name__ == '__main__':
    data_path = 'experimental/data'
    data = load_data(data_path)
    
    gemini = Gemini()
    
    strategy = SentenceEmbeddingStrategy(
        embed_model=gemini.set_model('models/text-embedding-004'),
        avoid_first=True
    )
    strategy.reimport_example(
        embed_dim=768,
        data=data[:1]
    )
    
    prompt = ExamplePrompt(type='md')
    gemini.set_model('models/gemini-2.5-pro-preview-03-25')
    for item in data:
        examples = []
        for example in strategy.get_example(item['text']):
            e_names = []
            for e in example['entities']:
                e_names.append(e['text'])
            examples.append({
                '输入': example,
                '输出': f"```json\n{{\"知识点\": {e_names}}}\n```"
            })
        # examples = [
        #     {
        #         '输入': example,
        #         '输出': "```json\n{\"知识点\": [\"知识点1\", \"知识点2\"]}\n```"
        #     }
        #     for example in strategy.get_example(item['text'])
        #     for e in example['entities']
        # ]
        message, instruction = prompt.get_ner_prompt(item['text'], examples)
        label = [(e['text'], e['type'][2:]) for e in item['entities']]
        gemini.instruction = instruction
        resp, _ = gemini.chat(message)
        resp = post_process(resp)
        pred = [(e, '知识点') for e in resp['知识点']]
        print(pred)
        print(label)
        break
