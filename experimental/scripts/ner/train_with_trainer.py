# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ner/train_with_trainer.py
# Description: 使用 Trainer 训练模型与评估

import os
import json
from random import shuffle
from transformers import (
    BertTokenizerFast, 
    BertConfig, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from model import BertBiLSTMCRF
from dataset import NERDataset
from config import *
from glob import glob
import swanlab
from datetime import datetime
from evaluate import load
from swanlab.integration.transformers import SwanLabCallback
from argparse import ArgumentParser
import math


def load_data(example_file):
    with open(example_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(eval_pred):
    pred_label_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
    
    ignore_mask = label_ids != 0
    pred_labels = [
        [id2label[id_] for id_ in pred_label_ids[idx][mask].tolist()]
        for idx, mask in enumerate(ignore_mask)
    ]

    labels = [
        [id2label[id_] for id_ in label_ids[idx][mask].tolist()]
        for idx, mask in enumerate(ignore_mask)
    ]

    metric = load("seqeval")
    results = metric.compute(predictions=pred_labels, references=labels)
    
    swanlab_results = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    swanlab.log(swanlab_results, print_to_console=True, zero_division=1)
    
    return results


def main(args):
    run_name = f"course_graph_ner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    swanlab_callback = SwanLabCallback(
        project="course_graph",
        name=run_name,
        config={
            "architecture": "BERT-BiLSTM-CRF",
            "optimizer": "AdamW",
            "scheduler": "linear_schedule_with_warmup",
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "bert_model": os.path.basename(args.bert_model_path)
        }
    )
    
    all_data = []
    for file in glob(args.data_path + '/*.json'):
        data = load_data(file)
        all_data.extend(data)
    shuffle(all_data)
    
    train_size = int(0.8 * len(all_data))
    val_size = int(math.ceil(0.1 * len(all_data)))
    train_data = all_data[:train_size]
    eval_data = all_data[train_size:train_size+val_size]
    test_data = all_data[train_size+val_size:]
    
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_path)
    train_dataset = NERDataset(train_data, tokenizer)
    eval_dataset = NERDataset(eval_data, tokenizer)
    test_dataset = NERDataset(test_data, tokenizer)
    
    config = BertConfig.from_pretrained(args.bert_model_path)
    config.num_labels = len(id2label)
    config._name_or_path = args.bert_model_path
    
    model = BertBiLSTMCRF(config)
    
    for param in model.bert.parameters():
        param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    swanlab_callback.update_config({
        "total_params": total_params, 
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params
    })
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.checkpoint,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=args.log,
        logging_steps=10,
        save_total_limit=3,  # 只保存最新的3个检查点
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        greater_is_better=True,
        no_cuda=False,
        fp16=True,      # 启用半精度训练
        dataloader_num_workers=4
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[swanlab_callback]
    )
    
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint, "final_model"))
    
    # test_results = trainer.evaluate(test_dataset)
    # swanlab.log(**test_results)
    swanlab.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='experimental/data')
    parser.add_argument("--bert_model_path", type=str, default='experimental/scripts/pre_trained/dienstag/chinese-bert-wwm-ext')
    parser.add_argument("--checkpoint", type=str, default='experimental/scripts/ner/checkpoints')
    parser.add_argument("--log", type=str, default='experimental/scripts/ner/logs')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    main(args)
