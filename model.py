import clip
import torch
import torch.nn.functional as F
from torch import nn


class FeatureFusion(nn.Module):
    """特征融合基类"""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def get_out_dim(self):
        return self.feature_dim


class ConcatFusion(FeatureFusion):
    """拼接融合"""

    def __init__(self, feature_dim):
        super().__init__(feature_dim)

    def forward(self, image_features, text_features):
        return torch.cat([image_features, text_features], dim=1)

    def get_out_dim(self):
        return self.feature_dim * 2


class LearnableWeightFusion(FeatureFusion):
    """可学习权重融合"""

    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        self.weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))

    def get_weights(self):
        return F.softmax(self.weights, dim=0)

    def forward(self, image_features, text_features):
        weights = self.get_weights()
        fused_features = weights[0] * image_features + weights[1] * text_features
        return fused_features


class AttentionFusion(FeatureFusion):
    """注意力融合"""

    def __init__(self, feature_dim, num_heads=8):
        super().__init__(feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

    def forward(self, image_features, text_features):
        image_features = image_features.float()
        text_features = text_features.float()

        # [B, 1, D]
        img_seq = image_features.unsqueeze(1)
        txt_seq = text_features.unsqueeze(1)
        # [B, 2, D]
        features = torch.cat([img_seq, txt_seq], dim=1)
        attn_output, _ = self.attention(features, features, features)
        return attn_output.mean(1)


class SentimentCLIP(nn.Module):
    """CLIP情感分类模型"""
    # CLIP模型特征维度映射
    CLIP_DIMS = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768
    }

    def __init__(self, clip_model="ViT-B/32", num_classes=3,
                 fusion_type="concat", ablation_mode=None):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model)

        # 获取特征维度
        self.feature_dim = self.CLIP_DIMS.get(clip_model, 512)

        # 消融实验模式
        self.ablation_mode = ablation_mode

        self.fusion = None
        # 选择融合方式
        if fusion_type == "concat":
            self.fusion = ConcatFusion(self.feature_dim)
        elif fusion_type == "weighted":
            self.fusion = LearnableWeightFusion(self.feature_dim)
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(self.feature_dim)

        # 构建分类器
        classifier_input_dim = (
            self.fusion.get_out_dim() if ablation_mode == 'none' else self.feature_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, num_classes)
        )

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.classifier = self.classifier.float()


    def forward(self, image, text):
        if self.ablation_mode == "image_only":
            features = self.clip_model.encode_image(image)
        elif self.ablation_mode == "text_only":
            features = self.clip_model.encode_text(text)
        else:
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            features = self.fusion(image_features, text_features)

        output = self.classifier(features.float())

        # 返回权重信息作为额外信息,而不是主要输出
        if isinstance(self.fusion, LearnableWeightFusion):
            self.current_weights = self.fusion.get_weights()

        return output
