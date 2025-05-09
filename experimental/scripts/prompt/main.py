# -*- coding: utf-8 -*-
# Create Date: 2025/04/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/main.py
# Description: 提示词测试


from course_graph.llm.prompt import (
    SentenceEmbeddingStrategy,
    ExamplePrompt,
    PromptStrategy,
    post_process
)
from course_graph import use_proxy
from course_graph.llm import Gemini, LLM, ONTOLOGY
from glob import glob
import json
from eval import evaluate
import swanlab
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import os
import sys


def load_data(path: str) -> list[dict]:
    return [
        line for file in glob(path + '/*.json') for line in json.load(open(file, 'r'))
    ]


def get_model_name(model) -> str:
    name = model.__class__.__name__
    if name == 'LLM':
        name = f'OpenAI(base_url={model.client.base_url})'
    return name + f'({model.model})'


def exp_ner(
        eval_data: list[dict],
        embed_model: LLM,
        chat_model: LLM,
        prompt_strategy: PromptStrategy,
        result_path: str = 'experimental/results/exp_ner_prompt',
        continue_file: str = None,
):
    run_name = f"course_graph_exp_ner_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []
    if continue_file:
        run_name = os.path.basename(continue_file).split('.')[0]
        results = json.load(open(continue_file, 'r'))
    
    swanlab.init(
        project="course_graph",
        experiment_name=run_name,
        mode="disabled" if sys.gettrace() is not None else "cloud",
        config={
            "prompt_strategy": prompt_strategy.__class__.__name__,
            "prompt_strategy_config": prompt_strategy.config,
            "embed_model": get_model_name(embed_model),
            "embed_model_config": embed_model.config,
            "chat_model": get_model_name(chat_model),
            "chat_model_config": chat_model.config,
        }
    )

    
    prompt = ExamplePrompt(type='md', strategy=prompt_strategy)
    
    if len(results) > 0:
        results.sort(key=lambda x: x['id'])
        eval_data = eval_data[results[-1]['id'] + 1:]
    try:
        for idx, item in enumerate(tqdm(eval_data)):
            message, instruction = prompt.get_ner_prompt(item['text'])
            label = [(e['text'], e['type']) for e in item['entities']]
            chat_model.instruction = instruction
            resp, _ = chat_model.chat(message)
            pred = []
            if chat_model.config['json']:
                resp = json.loads(resp)
            else:
                resp = post_process(resp)
            for k, v in resp.items():
                for n in v:
                    pred.append((n, k))
            eval_result = evaluate(pred, label)
            result = {
                'id': idx,
                'text': item['text'],
                'label': label,
                'pred': pred,
            }
            result.update(eval_result)
            results.append(result)
    except Exception as e:
        print(e)
        pass
    else:
        swanlab.log(pd.DataFrame(results)[['acc', 'precision', 'recall', 'f1']].mean().to_dict())
    finally:
        with open(os.path.join(result_path, f'{run_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        swanlab.finish()


if __name__ == '__main__':
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

    # strategy.reimport_example(data)

    exp_ner(
        eval_data=data,
        embed_model=text_embed,
        chat_model=llm,
        prompt_strategy=strategy
    )
