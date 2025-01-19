import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from dataset import get_train_val_loader
from evaluate import evaluate
from model import SentimentCLIP


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Single training epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader)
    for images, texts, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 如果是可学习权重融合,打印权重信息
        if hasattr(model, 'current_weights') and model.ablation_mode is None:
            pbar.set_description(f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}% '
                                 f'Weights: {model.current_weights.data.cpu().numpy()}')
        else:
            pbar.set_description(f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')

    return total_loss / len(dataloader), 100. * correct / total


def train(config):
    np.random.seed(config['data']['seed'])
    # Create model and move to device
    device = torch.device(config['training']['device'])
    model = SentimentCLIP(config['training']['clip_model'], fusion_type=config['training']['fusion'],
                          ablation_mode=config['training']['ablation']).to(device)

    print('Training with fusion:', config['training']['fusion'])
    print('Ablation mode:', config['training']['ablation'])

    # Create datasets and dataloaders
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = get_train_val_loader(config)

    df = pd.read_csv(config['data']['train_file'])
    class_counts = df['tag'].value_counts()
    total = len(df)

    # 计算初始权重
    initial_weights = total / (len(class_counts) * class_counts)

    # 使用平方根缓解极端权重
    weights = np.power(initial_weights, 0.3)

    # 归一化权重使其和为类别数量
    weights = weights / weights.sum() * len(class_counts)

    # 转换为tensor并移至设备
    weights = torch.FloatTensor([
        weights['negative'],
        weights['neutral'],
        weights['positive']
    ]).to(device)

    # Setup training
    if config['data']['imbalance_method'] == 'weighted':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=float(config['training']['learning_rate']),
                      weight_decay=float(config['training']['weight_decay']))

    # 学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(config['training']['learning_rate']),
        epochs=config['training']['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.04,  # 2 epochs预热
        anneal_strategy='cos'
    )

    # 早停设置
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    min_delta = 1e-4

    identifier = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    with open(f"./{config['training']['log_dir']}/log_{identifier}.txt", 'w') as f:
        f.write("Training log\n")
        f.write(f"Model: {config['training']['clip_model']}\n")
        f.write(f"Fusion: {config['training']['fusion']}\n")
        f.write(f"Ablation: {config['training']['ablation']}\n")

    # Training loop
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f'\nTraining epoch {epoch}')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer,
                                            scheduler, device)

        plot_dir = f"{config['training']['plot_dir']}/{identifier}_{epoch}_confusion_matrix.png"
        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device, is_test=False,
                                                                plot_dir=plot_dir)

        # 在config['training']['log_dir']中记录日志
        with open(f"./{config['training']['log_dir']}/log_{identifier}.txt", 'a') as f:
            f.write(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                    f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% Val Prec: {val_prec:.2f} Val Rec: {val_rec:.2f} Val F1: {val_f1:.2f}"
                    f" Patience: {patience_counter} lr: {scheduler.get_last_lr()[0]}\n")

        # 早停检查
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            },
                f"{config['training']['save_dir']}/{config['training']['fusion']}_{config['training']['ablation']}"
                f"_{config['data']['imbalance_method']}_best_model.pt")
        else:
            patience_counter += 1
            print('Patience counter:', patience_counter)
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
                f"{config['training']['save_dir']}/{config['training']['fusion']}_{config['training']['ablation']}"
                f"_{config['data']['imbalance_method']}_checkpoint_{epoch}.pt")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    os.makedirs(config['training']['plot_dir'], exist_ok=True)
    train(config)
