# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ke/train_with_trainer.py
# Description: 使用 Trainer 训练模型与评估

import os
import json
from random import shuffle
from transformers import (
    BertTokenizerFast, 
    BertConfig, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding
)
from model import BertBiLSTMCRF, BertForRE
from dataset import NERDataset, REDataset
from config import *
from glob import glob
import swanlab
from datetime import datetime
from evaluate import load
from swanlab.integration.transformers import SwanLabCallback
from argparse import ArgumentParser
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

seqeval = 'seqeval'
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data(example_file):
    with open(example_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_ner(eval_pred):
    pred_label_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
    
    valid_mask = pred_label_ids != -100

    pred_labels = [
        [id2label[id_] for id_ in pred_label_ids[idx][mask].tolist()]
        for idx, mask in enumerate(valid_mask)
    ]

    labels = [
        [id2label[id_] for id_ in label_ids[idx][mask].tolist()]
        for idx, mask in enumerate(valid_mask)
    ]

    global seqeval
    metric = load(seqeval)
    results = metric.compute(predictions=pred_labels, references=labels)
    return results


def compute_metrics_re(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return results


def main(args):
    
    config = {
        "ner": {
            "model": BertBiLSTMCRF,
            "compute_metrics": compute_metrics_ner,
            "dataset": NERDataset,
            "data_collator": DataCollatorForTokenClassification,
        },
        "re": {
            "model": BertForRE,
            "compute_metrics": compute_metrics_re,
            "dataset": REDataset,
            "data_collator": DataCollatorWithPadding,
        }
    }
    
    global seqeval
    if args.mode == "ner" and args.use_local_metric:  # 指定本地seqeval
        seqeval = args.seqeval_path
    
    model_config = BertConfig.from_pretrained(args.bert_model_path)
    model_config.num_labels = len(id2label)
    model_config._name_or_path = args.bert_model_path

    model = config[args.mode]["model"](model_config)

    run_name = f"course_graph_classical_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    swanlab.init(
        project="course_graph",
        experiment_name=run_name
    )
    swanlab_callback = SwanLabCallback()
    
    swanlab_callback.update_config({
        "architecture": model.__class__.__name__,
        "optimizer": "AdamW",
        "scheduler": "linear_schedule_with_warmup",
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "bert_model": os.path.basename(args.bert_model_path),
        "overflow_strategy": args.overflow_strategy
    })
    
    data = []
    for file in glob(args.data_path + '/*.json'):
        data.extend(load_data(file))
    shuffle(data)
    
    train_size = int(args.train_percent * len(data))
    eval_size = int(args.eval_percent * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:train_size+eval_size]
    test_data = data[train_size+eval_size:]
    
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_path)
    DatasetClass = config[args.mode]["dataset"]
    
    def create_dataset(data):
        return DatasetClass(
            data, 
            tokenizer, 
            args.max_len, 
            overflow_strategy=args.overflow_strategy,
        )
    
    train_dataset, eval_dataset, test_dataset = map(create_dataset, [train_data, eval_data, test_data])
    
    for param in model.bert.parameters():
        param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    swanlab_callback.update_config({
        "total_params": total_params, 
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "test_size": len(test_data),
    })
    
    DataCollatorClass = config[args.mode]["data_collator"]
    data_collator = DataCollatorClass(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=str(os.path.join(args.checkpoint, args.mode)),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.log, args.mode),
        logging_steps=10,
        save_total_limit=3,  # 只保存最新的3个检查点
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        greater_is_better=True,
        no_cuda=False,
        fp16=True,  # 启用半精度训练
        dataloader_num_workers=4
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=config[args.mode]["compute_metrics"],
        callbacks=[swanlab_callback]
    )
    
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint, "final_model"))
    
    test_results = trainer.evaluate(test_dataset)
    swanlab.log(test_results)
    swanlab.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="ner", choices=["ner", "re"])
    parser.add_argument("--data_path", type=str, default="experimental/data")    
    parser.add_argument("--train_percent", type=float, default=0.8)
    parser.add_argument("--eval_percent", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--overflow_strategy", type=str, default="truncation", choices=["truncation", "sliding_window"])
    parser.add_argument("--log", type=str, default="experimental/scripts/ke/logs")
    parser.add_argument("--bert_model_path", type=str, default="experimental/pre_trained/dienstag/chinese-bert-wwm-ext")
    parser.add_argument("--checkpoint", type=str, default="experimental/scripts/ke/checkpoints")
    parser.add_argument("--use_local_metric", type=bool, default=True)
    parser.add_argument("--seqeval_path", type=str, default="experimental/metrics/seqeval")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    main(args)
