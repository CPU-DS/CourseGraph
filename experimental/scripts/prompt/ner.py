# -*- coding: utf-8 -*-
# Create Date: 2025/04/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/ner.py
# Description: 实体识别


from course_graph.llm.prompt import (
    ExamplePrompt,
    PromptStrategy,
    post_process,
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
import traceback
from .metric import compute_overall_metrics, compute_sample_metrics


def get_model_name(model: LLM) -> str:
    name = model.__class__.__name__
    if name == 'LLM':
        name = f'OpenAI(base_url={model.client.base_url})'
    return name + f'({model.model})'


def main(
        eval_data: list[dict],
        embed_model: LLM,
        chat_model: LLM,
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
            "embed_model_config": embed_model.get_model_config(),
            "chat_model": get_model_name(chat_model),
            "chat_model_config": chat_model.get_model_config()
        }
    )

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
            if chat_model.config.get('json', False):
                resp = json.loads(resp)
            else:
                resp = post_process(resp)
            for k, v in resp.items():
                for n in v:
                    pred.append((n, k))
            eval_result = compute_sample_metrics(pred, label)
            result = {
                'id': idx,
                'text': item['text'],
                'label': label,
                'pred': pred,
            }
            result.update(eval_result)
            results.append(result)
    except Exception as e:
        traceback.print_exc()
    else:
        df = pd.DataFrame(results)
        labels = []
        for label in df['label']:
            labels.append([tuple(item) for item in label])
        preds = []
        for pred in df['pred']:
            preds.append([tuple(item) for item in pred])
        r = compute_overall_metrics(preds, labels)
        swanlab.log({
            'micro_precision': r['micro']['precision'],
            'micro_recall': r['micro']['recall'],
            'micro_f1': r['micro']['f1'],
            'precision': r['macro']['precision'],
            'recall': r['macro']['recall'],
            'f1': r['macro']['f1'],
            'accuracy': r['macro']['accuracy'],
        })
    finally:
        with open(os.path.join(result_path, f'{run_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        swanlab.finish()
