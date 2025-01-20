import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import get_test_loader
from model import SentimentCLIP


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
                                    zero_division=np.nan))
        report = classification_report(all_labels, all_predictions,
                                       target_names=['negative', 'neutral', 'positive'],
                                       digits=4,
                                       zero_division=np.nan,
                                       output_dict=True)
        weighted_f1 = report['weighted avg']['f1-score']
        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']

        return total_loss / len(dataloader), 100. * correct / total, weighted_precision, weighted_recall, weighted_f1

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
    print("Model loaded from checkpoint")
    print("fusion_type:", config['training']['fusion'])
    print("ablation_mode:", config['training']['ablation'])

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
    results_df.to_csv(f'predictions_{config['training']['fusion']}.txt', index=False)
    print("Predictions saved to predictions.txt")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs(config['training']['save_dir'], exist_ok=True)
    set_seed(config['data']['seed'])
    test(config)
