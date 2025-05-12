# -*- coding: utf-8 -*-
# Create Date: 2025/04/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/ner.py
# Description: 实体识别


from course_graph.llm.prompt import (
    ExamplePrompt,
    PromptStrategy,
    post_process,
    Filter
)
from course_graph.llm import LLM
import json
import swanlab
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import os
import sys
from typing import Optional


def metric(pred, label):
    # pred/label 为 list of (text, type)
    pred_set = set(pred)
    label_set = set(label)
    correct = len(pred_set & label_set)

    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(label_set) if label_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    union_size = len(pred_set | label_set)
    acc = correct / union_size if union_size > 0 else 1.0

    return {
        'precision': precision,
        'recall': recall,
        'acc': acc,
        'f1': f1
    }


def get_model_name(model) -> str:
    name = model.__class__.__name__
    if name == 'LLM':
        name = f'OpenAI(base_url={model.client.base_url})'
    return name + f'({model.model})'


def main(
        eval_data: list[dict],
        embed_model: LLM,
        chat_model: LLM,
        reimport_example: bool = False,
        example_filter: Optional[Filter] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        result_path: str = 'experimental/results/exp_ner_prompt',
        continue_file: Optional[str] = None,
):
    """
    评估提示词进行命名实体识别

    Args:
        eval_data (list[dict]): 需要评估的数据
        embed_model (LLM): 嵌入模型
        chat_model (LLM): 聊天模型
        reimport_example (bool, optional): 是否重新导入示例. Defaults to False.
        example_filter (Filter, optional): 过滤策略. Defaults to None.
        prompt_strategy (PromptStrategy, optional): 提示词策略. Defaults to None.
        result_path (str, optional): 结果保存路径. Defaults to 'experimental/results/exp_ner_prompt'.
        continue_file (str, optional): 继续上次运行的文件. Defaults to None.
    """
    run_name = f"course_graph_exp_ner_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []
    if continue_file and os.path.exists(continue_file):
        run_name = os.path.basename(continue_file).split('.')[0]
        results = json.load(open(continue_file, 'r'))

    swanlab.init(
        project="course_graph",
        experiment_name=run_name,
        mode="disabled" if sys.gettrace() is not None else "cloud",
        config={
            "prompt_strategy": prompt_strategy.__class__.__name__ if prompt_strategy else "None",
            "prompt_strategy_config": prompt_strategy.config if prompt_strategy else "None",
            "embed_model": get_model_name(embed_model),
            "embed_model_config": embed_model.config,
            "chat_model": get_model_name(chat_model),
            "chat_model_config": chat_model.config,
            "filter": example_filter.__class__.__name__ if example_filter else "None",
            "filter_config": example_filter.config if example_filter else "None",
        }
    )
    
    if reimport_example:
        if example_filter is not None:
            prompt_strategy.reimport_example(example_filter.filter(eval_data))
        else:
            prompt_strategy.reimport_example(eval_data)

    prompt = ExamplePrompt(type='md', strategy=prompt_strategy)

    skip = 0
    if len(results) > 0:
        skip = results[-1]['id'] + 1
    try:
        for idx, item in enumerate(tqdm(eval_data)):
            if idx < skip:
                continue
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
            eval_result = metric(pred, label)
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
