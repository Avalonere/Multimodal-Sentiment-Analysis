import os

import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class MultiModalDataset(Dataset):
    """Construct the dataset"""

    def __init__(self, df: pd.DataFrame, img_dir: str, txt_dir: str,
                 transform=None, has_label: bool = True,
                 device: str = "cuda"):
        self.df = df
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.transform = transform
        self.has_label = has_label
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        guid = str(int(self.df.iloc[idx]['guid']))
        img_path = os.path.join(self.img_dir, f"{guid}.jpg")
        txt_path = os.path.join(self.txt_dir, f"{guid}.txt")

        # Load and process image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load and tokenize text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        if self.has_label:
            label = torch.tensor(self.label_map[self.df.iloc[idx]['tag']],
                                 device=self.device)
            return image.to(self.device), text, label
        return image.to(self.device), text, guid


def collate_fn(batch):
    """Custom collate function to handle text tokenization"""
    # Unzip the batch
    images, texts, labels = zip(*batch)

    # Stack images (already tensors)
    images = torch.stack(images)

    # Process texts - ensure they are strings before tokenization
    texts = [str(text) for text in texts]
    text_tokens = clip.tokenize(texts, truncate=True).to(images.device)

    # Stack labels if they are tensors
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)

    return images, text_tokens, labels


def get_train_val_loader(config):
    """Get train and validation dataloaders"""
    _, preprocess = clip.load(config['training']['clip_model'])
    device = config['training']['device']

    model = config['training']['clip_model']
    size = 224
    if '336' in model:
        size = 336
    train_transform = transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1),
                                interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    df = pd.read_csv(config['data']['train_file'])

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config['data']['val_size'],
        random_state=config['data']['seed']
    )

    for train_idx, val_idx in splitter.split(df, df['tag']):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

    # 随机抽样
    # train_df = df.sample(frac=0.875, random_state=config['data']['seed'])
    # val_df = df.drop(train_df.index)

    # 计算类别权重
    class_counts = train_df['tag'].value_counts().sort_index()  # 确保顺序一致
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    # 归一化权重
    weights = weights / np.sum(weights) * len(class_counts)
    sample_weights = [weights[train_df.iloc[i]['tag']] for i in range(len(train_df))]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True
    )

    train_dataset = MultiModalDataset(
        train_df,
        config['data']['img_dir'],
        config['data']['txt_dir'],
        transform=train_transform,
        device=device
    )

    val_dataset = MultiModalDataset(
        val_df,
        config['data']['img_dir'],
        config['data']['txt_dir'],
        transform=preprocess,
        device=device,
    )

    if config['data']['imbalance_method'] == 'sample':
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['training']['batch_size'],
                                  sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['training']['batch_size'],
                                  shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


def visualize_label_distribution(csv_path: str, save_path: str = None):
    """Visualize label distribution using seaborn"""
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    sns.set_theme()
    sns.countplot(data=df, x='tag', palette='husl', hue='tag', legend=False)
    plt.title('Distribution of Sentiment Labels')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # print the distribution
    print(df['tag'].value_counts())


def get_test_loader(config) -> DataLoader:
    """Get test dataloader"""
    # Load CLIP preprocessing
    _, preprocess = clip.load(config['training']['clip_model'])

    # Read test data
    df = pd.read_csv(config['data']['test_file'])

    device = config['training']['device']

    # Create dataset
    test_dataset = MultiModalDataset(
        df,
        config['data']['img_dir'],
        config['data']['txt_dir'],
        transform=preprocess,
        has_label=False,
        device=device,
    )

    # Create loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Test: {len(df)} samples")
    return test_loader


if __name__ == "__main__":
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Test train/val loaders
    train_loader, val_loader = get_train_val_loader(config)
    test_loader = get_test_loader(config)

    # Visualize distribution
    visualize_label_distribution(config['data']['train_file'], save_path='label_distribution.png')
