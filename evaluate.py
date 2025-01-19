import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm

from dataset import get_test_loader
from model import SentimentCLIP


def evaluate(model, dataloader, criterion, device, is_test=False, plot_dir=None):
    """通用评估函数"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    predictions = []
    guids = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluating'):
            images, texts, labels_or_guids = data
            images = images.to(device)

            outputs = model(images, texts)

            if not is_test:
                labels = labels_or_guids.to(device)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 收集所有预测和标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
                guids.extend(labels_or_guids)

    # 计算混淆矩阵和分类报告
    if not is_test:
        print("Validation Metrics:")
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_predictions))
        # seaborn作图并保存
        sns.heatmap(confusion_matrix(all_labels, all_predictions), annot=True, fmt='d',
                    cmap=sns.light_palette("seagreen", as_cmap=True))
        # title
        plt.title('Confusion Matrix')
        # x轴标签
        plt.xlabel('Predicted')
        # y轴标签
        plt.ylabel('True')
        # 把0,1,2改成negative, neutral, positive
        plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['negative', 'neutral', 'positive'])
        plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['negative', 'neutral', 'positive'])
        plt.tight_layout()
        plt.savefig(plot_dir)
        plt.close()

        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions,
                                    target_names=['negative', 'neutral', 'positive'],
                                    digits=4,
                                    zero_division=0))

        return (total_loss / len(dataloader), 100. * correct / total,
                precision_score(all_labels, all_predictions, average='weighted', zero_division=np.nan),
                recall_score(all_labels, all_predictions, average='weighted', zero_division=np.nan),
                f1_score(all_labels, all_predictions, average='weighted', zero_division=np.nan))

    else:
        return predictions, guids


def test(config):
    """测试集预测"""
    device = torch.device(config['training']['device'])
    model = SentimentCLIP(config['training']['clip_model'], fusion_type=config['training']['fusion'],
                          ablation_mode=config['training']['ablation'], ).to(device)

    # 加载最佳模型
    checkpoint = torch.load(
        f"{config['training']['save_dir']}/{config['training']['fusion']}_{config['training']['ablation']}"
        f"_{config['data']['imbalance_method']}_best_model.pt",
        weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取测试集
    test_loader = get_test_loader(config)

    # 预测
    predictions, guids = evaluate(model, test_loader, None, device, is_test=True)

    # 转换预测结果
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predictions = [label_map[p] for p in predictions]

    # 保存结果
    results_df = pd.DataFrame({
        'guid': guids,
        'tag': predictions
    })
    results_df.to_csv('predictions.txt', index=False)
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs(config['training']['save_dir'], exist_ok=True)

    test(config)
