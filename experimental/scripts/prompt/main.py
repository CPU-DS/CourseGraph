# -*- coding: utf-8 -*-
# Create Date: 2025/04/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/main.py
# Description: 提示词测试


from course_graph.llm.prompt import (
    SentenceEmbeddingStrategy,
    ExamplePrompt
)
import json
from course_graph import use_proxy
from course_graph.llm import Gemini
from glob import glob
import json
from eval import evaluate
import swanlab
from datetime import datetime
import pandas as pd
from tqdm import tqdm


use_proxy()


def load_data(path: str) -> list[dict]:
    return [
        line for file in glob(path + '/*.json') for line in json.load(open(file, 'r'))
    ]


if __name__ == '__main__':
    data_path = 'experimental/data'
    data = load_data(data_path)

    embed_model = Gemini()
    embed_model.model = 'text-embedding-004'
    
    llm = Gemini()
    llm.model = 'gemini-2.5-flash-preview-04-17'
    llm.config['reasoning_effort'] = 'high'
    llm.config['json'] = True

    run_name = f"course_graph_prompt_experimental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    swanlab.init(
        project="course_graph",
        experiment_name=run_name,
        config={
                "strategy": "sentence_embedding",
                "topk": 3,
                "avoid_first": True,
                "data_path": data_path,
                "embed_model": embed_model.model,
                "embed_dim": 768,
                "embed_model_config": embed_model.config,
                "chat_model": llm.model,
                "chat_model_config": llm.config,
            }
    )
    
    strategy = SentenceEmbeddingStrategy(
        embed_model=embed_model,
        embed_dim=swanlab.config['embed_dim'],
        avoid_first=swanlab.config['avoid_first'],
        topk=swanlab.config['topk'],
    )
    strategy.json_block = False

    # strategy.reimport_example(
    #     data=data
    # )
    
    eval_result = []
    prompt = ExamplePrompt(type='md', strategy=strategy)
    for item in tqdm(data):
        message, instruction = prompt.get_ner_prompt(item['text'])
        label = [(e['text'], e['type'][2:]) for e in item['entities']]
        llm.instruction = instruction
        resp, _ = llm.chat(message)
        resp = json.loads(resp)
        pred = [(e, '知识点') for e in resp['知识点']]
        eval_result.append(evaluate(pred, label))

    swanlab.log(pd.DataFrame(eval_result).mean().to_dict())
    
    
