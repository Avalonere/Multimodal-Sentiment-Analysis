import glob
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 读取所有日志文件
log_files = glob.glob('./logs/*.txt')
data = []
sns.set_theme()
for file in log_files:
    with open(file, 'r') as f:
        content = f.read()

        # 提取fusion和ablation信息
        fusion = re.search(r'Fusion: (\w+)', content)
        ablation = re.search(r'Ablation: (\w+)', content)

        if fusion and ablation:
            fusion = fusion.group(1)
            ablation = ablation.group(1)

            # 提取每个epoch的指标
            epochs = re.findall(r'Epoch \d+ .+?weights: .*?\n', content)
            for epoch in epochs:
                epoch_num = int(re.search(r'Epoch (\d+)', epoch).group(1))
                train_loss = float(re.search(r'Train Loss: ([\d\.]+)', epoch).group(1))
                val_loss = float(re.search(r'Val Loss: ([\d\.]+)', epoch).group(1))
                val_prec = float(re.search(r'Val Prec: ([\d\.]+)', epoch).group(1))
                val_rec = float(re.search(r'Val Rec: ([\d\.]+)', epoch).group(1))
                val_f1 = float(re.search(r'Val F1: ([\d\.]+)', epoch).group(1))

                # 提取weights (如果有)
                weights_match = re.search(r'weights: \[([\d\. ]+)\]', epoch)
                weights = None if weights_match is None else [float(w) for w in weights_match.group(1).split()]

                data.append({
                    'fusion': fusion,
                    'ablation': ablation,
                    'epoch': epoch_num,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_prec': val_prec,
                    'val_rec': val_rec,
                    'val_f1': val_f1,
                    'weights': weights
                })

df = pd.DataFrame(data)

# 添加fusion映射字典
fusion_mapping = {
    'concat': 'Concat',
    'weighted': 'Weighted',
    'attention': 'Attention',
    'attention_alt': 'Attention_Alt',
    'text_only': 'Text_Only',
    'image_only': 'Image_Only',
    'none': ''  # 对于消融实验,fusion显示为空
}

# 1. 修改第一张图的代码
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
palette = sns.husl_palette(n_colors=len(df[['fusion', 'ablation']].drop_duplicates()))

for (fusion, ablation), color in zip(df[['fusion', 'ablation']].drop_duplicates().values, palette):
    mask = (df['fusion'] == fusion) & (df['ablation'] == ablation)
    # 修改label显示
    if ablation == 'none':
        label = fusion_mapping[fusion]
    else:
        label = ablation.capitalize()

    sns.lineplot(data=df[mask], x='epoch', y='val_prec', ax=ax1, label=label, color=color)
    sns.lineplot(data=df[mask], x='epoch', y='val_rec', ax=ax2, label=label, color=color)
    sns.lineplot(data=df[mask], x='epoch', y='val_f1', ax=ax3, label=label, color=color)

ax1.set_title('Precision')
ax2.set_title('Recall/Accuracy')
ax3.set_title('F1')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Precision')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Recall/Accuracy')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('F1')
plt.tight_layout()
plt.show()

# 2. 修改loss对比图,val loss右移
husl_colors = sns.husl_palette(3)  # 获取3色husl色板
concat_none = df[(df['fusion'] == 'concat') & (df['ablation'] == 'none')]

plt.figure(figsize=(8, 5))
sns.lineplot(data=concat_none, x='epoch', y='train_loss', label='Train', color=husl_colors[0])
sns.lineplot(data=concat_none, x='epoch', y='val_loss', label='Val', color=husl_colors[-1])
plt.title('Training and Validation Loss (Fusion: concat, Ablation: none)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# 3. 修改weights变化图配色
weighted_none = df[(df['fusion'] == 'weighted') & (df['ablation'] == 'none')]
if len(weighted_none) > 0 and weighted_none['weights'].iloc[0] is not None:
    plt.figure(figsize=(8, 5))
    image_weights = [w[0] for w in weighted_none['weights']]
    text_weights = [w[1] for w in weighted_none['weights']]

    sns.lineplot(x=weighted_none['epoch'], y=image_weights, label='Image', color=husl_colors[0])
    sns.lineplot(x=weighted_none['epoch'], y=text_weights, label='Text', color=husl_colors[-1])
    plt.title('Modality Weights Evolution (Fusion: weighted, Ablation: none)')

    # x轴标签
    plt.xlabel('Epoch')
    # y轴标签
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()
