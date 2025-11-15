# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config  # 导入配置

def get_data_loaders():
    """
    根据config.py中的配置，创建并返回训练集和测试集的数据加载器。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    train_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    print(f"Data loaded successfully. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return train_loader, test_loader