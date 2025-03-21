import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
from model import BertBiLSTMCRF
from dataset import NERDataset
from config import *
from glob import glob
import swanlab
from datetime import datetime

model_path = 'experimental/scripts/pre_trained/dienstag/chinese-bert-wwm-ext'
data_path = 'experimental/data'
checkpoint_dir = 'experimental/scripts/ner/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)


def load_data(example_file):
    with open(example_file, "r", encoding="utf-8") as f:
        return json.load(f)


def train(model, train_dataloader, optimizer, scheduler, device, num_epochs=5, save_steps=50):
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            global_step += 1
            swanlab.log({
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/global_step": global_step
            })
            
            if global_step % save_steps == 0:
                save_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, save_path)
        
        epoch_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss/len(train_dataloader),
        }, epoch_save_path)
        
        epoch_loss = total_loss/len(train_dataloader)
        swanlab.log({
            "train/epoch": epoch + 1,
            "train/epoch_loss": epoch_loss
        }, print_to_console=True)


def main():
    run_name = f"course_graph_ner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = swanlab.init(
        project="course_graph",
        name=run_name,
        config='experimental/scripts/ner/swanlab-init-config.yaml'
    )
    
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertBiLSTMCRF(model_path, run.config['labels']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())  # 总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量
    run.config.update({"total_params": total_params, "trainable_params": trainable_params})
    
    all_data = []
    for file in glob(data_path + '/*.json'):
        data = load_data(file)
        all_data.extend(data)
    
    dataset = NERDataset(all_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=run.config['batch_size'], shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=run.config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(dataloader) * run.config['epochs']
    )
    
    train(model, dataloader, optimizer, scheduler, device, num_epochs=run.config['epochs'], save_steps=100)
    
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    swanlab.finish()


if __name__ == "__main__":
    main()
