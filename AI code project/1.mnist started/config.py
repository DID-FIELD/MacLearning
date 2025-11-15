# config.py

import torch

# --- 设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 数据配置 ---
DATA_DIR = './data'
BATCH_SIZE = 64
# MNIST的均值和标准差（预处理用）
NORMALIZE_MEAN = (0.1307,)
NORMALIZE_STD = (0.3081,)

# --- 模型配置 ---
# 可以在这里定义不同的模型配置，方便切换
MODEL_CONFIG = {
    'type': 'SimpleMNISTModel',  # 选择要使用的模型类
    'hidden_size1': 128,         # 第一个隐藏层大小
    'hidden_size2': 32,          # 第二个隐藏层大小
    'activation1': 'ReLU',       # 第一个激活函数
    'activation2': 'LeakyReLU'   # 第二个激活函数
}

# --- 训练配置 ---
EPOCHS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# --- 保存与日志配置 ---
RESULTS_DIR = './results'
SAVE_MODEL = True
MODEL_SAVE_PATH = f'{RESULTS_DIR}/best_model.pth'
PLOT_RESULTS = True