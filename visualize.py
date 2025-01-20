import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm import tqdm

from model import SentimentCLIP

plt.style.use('ggplot')
sns.set_theme()
sns.set_palette('husl')


def load_best_model(concat=False, weighted=False, attention=False, attention_alt=False, text_only=False,
                    image_only=False):
    """
    加载最佳模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if concat:
        model_concat = SentimentCLIP("ViT-L/14@336px", fusion_type="concat", ablation_mode="none")
        checkpoint_concat = torch.load('checkpoints/concat_none_weighted_best_model.pt', weights_only=False)
        model_concat.load_state_dict(checkpoint_concat['model_state_dict'])
        model_concat.to(device)
        model_concat.eval()
        return model_concat, device
    if weighted:
        model_weighted = SentimentCLIP("ViT-L/14@336px", fusion_type="weighted", ablation_mode="none")
        checkpoint_weighted = torch.load('checkpoints/weighted_none_weighted_best_model.pt', weights_only=False)
        model_weighted.load_state_dict(checkpoint_weighted['model_state_dict'])
        model_weighted.to(device)
        model_weighted.eval()
        return model_weighted, device
    if attention:
        model_attention = SentimentCLIP("ViT-L/14@336px", fusion_type="attention", ablation_mode="none")
        checkpoint_attention = torch.load('checkpoints/attention_none_weighted_best_model.pt', weights_only=False)
        model_attention.load_state_dict(checkpoint_attention['model_state_dict'])
        model_attention.to(device)
        model_attention.eval()
        return model_attention, device
    if attention_alt:
        model_attention_alt = SentimentCLIP("ViT-L/14@336px", fusion_type="attention_alt", ablation_mode="none")
        checkpoint_attention_alt = torch.load('checkpoints/attention_alt_none_weighted_best_model.pt',
                                              weights_only=False)
        model_attention_alt.load_state_dict(checkpoint_attention_alt['model_state_dict'])
        model_attention_alt.to(device)
        model_attention_alt.eval()
        return model_attention_alt, device
    if text_only:
        model_text_only = SentimentCLIP("ViT-L/14@336px", fusion_type="none", ablation_mode="text_only")
        checkpoint_text_only = torch.load('checkpoints/none_text_only_weighted_best_model.pt', weights_only=False)
        model_text_only.load_state_dict(checkpoint_text_only['model_state_dict'])
        model_text_only.to(device)
        model_text_only.eval()
        return model_text_only, device
    if image_only:
        model_image_only = SentimentCLIP("ViT-L/14@336px", fusion_type="none", ablation_mode="image_only")
        checkpoint_image_only = torch.load('checkpoints/none_image_only_weighted_best_model.pt', weights_only=False)
        model_image_only.load_state_dict(checkpoint_image_only['model_state_dict'])
        model_image_only.to(device)
        model_image_only.eval()
        return model_image_only, device


def analyze_confidence(model, val_loader, device, name):
    """分析预测置信度分布"""
    sns.set_palette('husl', 3)
    confidences = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for images, texts, labels in tqdm(val_loader):
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            probs = F.softmax(outputs, dim=1)

            for label, prob in zip(labels, probs):
                confidences[label.item()].append(prob.max().item())

    plt.figure(figsize=(10, 6))
    for label in [0, 1, 2]:
        sns.kdeplot(confidences[label], label=f'Class {label}')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    # 把图例上的0,1,2改成negative, neutral, positive
    plt.gca().get_legend().get_texts()[0].set_text('Negative')
    plt.gca().get_legend().get_texts()[1].set_text('Neutral')
    plt.gca().get_legend().get_texts()[2].set_text('Positive')
    plt.tight_layout()
    plt.savefig(f'vis/confidence_dist_{name}.png')
    plt.close()


def analyze_errors(model, val_loader, device, name):
    """分析错误预测案例"""
    errors = []

    denorm = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
    )

    with torch.no_grad():
        for images, texts, labels in tqdm(val_loader):
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            _, predicted = outputs.max(1)

            # 记录错误预测
            mask = predicted != labels
            if mask.any():
                wrong_idx = mask.nonzero().squeeze()
                for idx in wrong_idx:
                    # 先进行反归一化
                    img_denorm = denorm(images[idx].cpu())
                    # 裁剪到0-1范围
                    img_denorm = torch.clamp(img_denorm, 0, 1)

                    errors.append({
                        'image': img_denorm,
                        'text': texts[idx],
                        'true': labels[idx].item(),
                        'pred': predicted[idx].item(),
                        'conf': F.softmax(outputs[idx], dim=0).max().item()
                    })

            if len(errors) >= 10:  # 只保存前10个错误案例
                break

    errors = errors[:10]
    # 保存错误案例
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, error in enumerate(errors):
        ax = axes[i // 5, i % 5]
        ax.imshow(error['image'].permute(1, 2, 0))
        ax.set_title(f'True: {error["true"]}\nPred: {error["pred"]}\nConf: {error["conf"]:.2f}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'vis/error_cases_{name}.png')
    plt.close()


def visualize_features(model, dataloader, device, name):
    """
    可视化特征
    """
    # 收集特征和标签
    fused_features = []
    hidden_features = []
    labels = []
    sns.set_palette('husl', 3)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, texts, batch_labels = [x.to(device) for x in batch]

            # 获取融合特征
            if model.ablation_mode == "image_only":
                features = model.clip_model.encode_image(images)
            elif model.ablation_mode == "text_only":
                features = model.clip_model.encode_text(texts)
            else:
                image_features = model.clip_model.encode_image(images)
                text_features = model.clip_model.encode_text(texts)
                features = model.fusion(image_features, text_features)

            fused_features.append(features.cpu())

            # 获取隐藏层特征(如果不是attention_alt)
            if name != 'attention_alt' and not isinstance(model.classifier, nn.Linear):
                hidden = model.classifier[0](features.float())
                hidden = model.classifier[1](hidden)
                hidden = model.classifier[2](hidden)
                hidden_features.append(hidden.cpu())

            labels.append(batch_labels.cpu())

    fused_features = torch.cat(fused_features).numpy()
    labels = torch.cat(labels).numpy()

    # 创建降维模型
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    reducer = umap.UMAP()

    # 对融合特征进行降维可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Fused Features Visualization")

    embeddings = {
        'PCA': pca.fit_transform(fused_features),
        'T-SNE': tsne.fit_transform(fused_features),
        'UMAP': reducer.fit_transform(fused_features)
    }

    for idx, (title, embedding) in enumerate(embeddings.items()):
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=labels,
            palette=sns.husl_palette(3),
            ax=axes[idx]
        )
        axes[idx].set_title(title)

        # 把图例上的0,1,2改成negative, neutral, positive
        axes[idx].get_legend().get_texts()[0].set_text('Negative')
        axes[idx].get_legend().get_texts()[1].set_text('Neutral')
        axes[idx].get_legend().get_texts()[2].set_text('Positive')

    plt.tight_layout()
    plt.savefig(f'vis/fused_features_{name}.png')
    plt.close()

    # 如果不是attention_alt,还要对隐藏层特征可视化
    if name != 'attention_alt' and not isinstance(model.classifier, nn.Linear):
        hidden_features = torch.cat(hidden_features).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Hidden Layer Features Visualization")

        embeddings = {
            'PCA': pca.fit_transform(hidden_features),
            'T-SNE': tsne.fit_transform(hidden_features),
            'UMAP': reducer.fit_transform(hidden_features)
        }

        for idx, (title, embedding) in enumerate(embeddings.items()):
            sns.scatterplot(
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=labels,
                palette=sns.husl_palette(3),
                ax=axes[idx]
            )
            axes[idx].set_title(title)
            # 把图例上的0,1,2改成negative, neutral, positive
            axes[idx].get_legend().get_texts()[0].set_text('Negative')
            axes[idx].get_legend().get_texts()[1].set_text('Neutral')
            axes[idx].get_legend().get_texts()[2].set_text('Positive')

        plt.tight_layout()
        plt.savefig(f'vis/hidden_features_{name}.png')
        plt.close()


def visualize_attention_weights(model, dataloader, device, name):
    """可视化注意力权重"""

    # 收集注意力权重
    attention_weights = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, texts, _ = [x.to(device) for x in batch]

            # 获取注意力权重
            image_features = model.clip_model.encode_image(images)
            text_features = model.clip_model.encode_text(texts)

            # [B,1,D]
            img_seq = image_features.unsqueeze(1).float()
            txt_seq = text_features.unsqueeze(1).float()
            # [B,2,D]
            features = torch.cat([img_seq, txt_seq], dim=1)

            # 获取注意力权重 [B,2,2]
            _, weights = model.fusion.attention(features, features, features, need_weights=True)
            attention_weights.append(weights.cpu())

            # 只取一个batch的数据
            break

    # 计算平均注意力权重
    avg_attention = torch.mean(torch.cat(attention_weights), dim=0)

    # 创建热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        avg_attention.numpy(),
        annot=True,
        fmt='.3f',
        xticklabels=['Image', 'Text'],
        yticklabels=['Image', 'Text'],
        cmap=sns.color_palette("light:b", as_cmap=True)
    )
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    plt.savefig(f'vis/attention_weights_{name}.png')
    plt.close()


def analyze_model_behaviors(model, dataloader, device, name):
    """综合分析模型行为"""
    activations = {'fc1': []}
    class_scores = {0: [], 1: [], 2: []}

    # 修改hook函数
    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())

        return hook

    # 注册hooks
    if not isinstance(model.classifier, nn.Linear):
        model.classifier[0].register_forward_hook(get_activation('fc1'))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, texts, labels = [x.to(device) for x in batch]

            outputs = model(images, texts)
            probs = F.softmax(outputs, dim=1)

            for i, label in enumerate(labels):
                class_scores[label.item()].append(probs[i].cpu().numpy())

    # 1. 绘制激活分布
    if not isinstance(model.classifier, nn.Linear):
        plt.figure(figsize=(6, 4))
        act = torch.cat(activations['fc1']).flatten()
        sns.kdeplot(act.numpy())
        plt.title('fc1 Distribution')
        plt.tight_layout()
        plt.savefig(f'vis/activations_{name}.png')
        plt.close()

    #  类别预测模式
    plt.figure(figsize=(12, 4))
    map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    for i in range(3):
        scores = np.array(class_scores[i])
        plt.subplot(1, 3, i + 1)
        sns.heatmap(scores[:min(50, len(scores))], cmap='YlOrRd')
        # x轴把012改成negative, neutral, positive
        plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['Negative', 'Neutral', 'Positive'])
        plt.title(f'Class {map.get(i)} Prediction Patterns')
    plt.tight_layout()
    plt.savefig(f'vis/class_patterns_{name}.png')
    plt.close()


# if __name__ == '__main__':
#     names = ['concat', 'weighted', 'attention', 'attention_alt', 'text_only', 'image_only']
#     for name in names:
#         model, device = load_best_model(**{name: True})
#         with open("config.yaml", 'r') as f:
#             config = yaml.safe_load(f)

#         _, val_loader = get_train_val_loader(config)

#         # 创建可视化目录
#         os.makedirs('vis', exist_ok=True)

#         # 运行可视化分析
#         analyze_confidence(model, val_loader, device, name)
#         analyze_errors(model, val_loader, device, name)
#         visualize_features(model, val_loader, device, name)
#         analyze_model_behaviors(model, val_loader, device, name)
#         if name == 'attention' or name == 'attention_alt':
#             visualize_attention_weights(model, val_loader, device, name)


class GradCAM:
    """
    Grad-CAM 算法
    """
    def __init__(self, model, target_block):
        self.model = model
        self.block = target_block
        self.gradients = None
        self.activation = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activation = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.block.register_forward_hook(forward_hook)
        self.block.register_backward_hook(backward_hook)

    def __call__(self, x, target_idx=None):
        x = x.requires_grad_(True)

        # 获取完整的模型输出
        image_features = self.model.clip_model.encode_image(x)
        image_features = image_features.float()
        # 获取分类logits
        logits = self.model.classifier(image_features)

        if target_idx is None:
            target_idx = logits.argmax(dim=1)

        # 使用交叉熵损失
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([target_idx], device=x.device)
        loss = criterion(logits, target)

        self.model.zero_grad()
        loss.backward()

        # 生成CAM部分保持不变
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(alpha * self.activation, dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def gradcam_for_single_image(model, image_path, target_id, device):
    """
    对单张图像执行 Grad-CAM
    """
    # 图像预处理
    _, preprocess = clip.load('ViT-L/14@336px')
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 将 CLIP ViT 的 patch embedding 层视为 target_block（以ViT-L/14为例，如需更换层名请自行修改）
    target_block = model.clip_model.visual.conv1

    # 启用临时梯度计算
    for param in target_block.parameters():
        param.requires_grad = True

    cam_generator = GradCAM(model, target_block)
    cam = cam_generator(image_tensor, target_id)

    # 还原冻结状态
    for param in target_block.parameters():
        param.requires_grad = False

    # 返回热力图
    return cam


def overlay_cam_on_image(img_path, cam):
    """
    将 Grad-CAM 热力图叠加到原图上
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # 确保cam不为None也没有NaN
    cam = np.nan_to_num(cam, nan=0.0)
    cam = cam.astype(np.float32)

    # 将cam缩放到与图像相同的大小 (注意cv2.resize的形参顺序是(宽, 高))
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

    cam_resized = (cam_resized * 255).astype(np.uint8)

    # 转为伪彩色
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    # 叠加
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return overlay


if __name__ == '__main__':
    # 简例示范：只以 'image_only' 下的模型为例
    model, device = load_best_model(image_only=True)
    model.eval()
    cam_map = gradcam_for_single_image(model, './data/data/231.jpg', 2, device)
    # 这里可以进一步绘制或保存 cam_map
    print("Grad-CAM 计算完成:", cam_map.shape)

    # 用法示例
    overlayed_img = overlay_cam_on_image('./data/data/231.jpg', cam_map)
    cv2.imwrite('./vis/gradcam_overlay.jpg', overlayed_img)
