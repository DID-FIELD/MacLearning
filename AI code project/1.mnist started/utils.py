# utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import config

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建。"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_and_save_results(results):
    """
    根据训练结果绘制并保存图表。
    """
    ensure_dir(config.RESULTS_DIR)
    
    epochs = range(1, config.EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='Training Loss')
    plt.plot(epochs, results['test_loss'], label='Test Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='Training Accuracy')
    plt.plot(epochs, results['test_acc'], label='Test Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(config.RESULTS_DIR, 'training_curves.png')
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    if config.PLOT_RESULTS:
        plt.show()
    plt.close()

def save_results(results):
    """
    保存训练结果字典到文件。
    """
    ensure_dir(config.RESULTS_DIR)
    results_path = os.path.join(config.RESULTS_DIR, 'training_results.npy')
    np.save(results_path, results)
    print(f"Training results saved to {results_path}")

def save_model(model, epoch, test_acc):
    """
    保存模型 checkpoint。
    """
    if config.SAVE_MODEL:
        ensure_dir(config.RESULTS_DIR)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'test_accuracy': test_acc,
            'config': config.MODEL_CONFIG # 同时保存模型配置，便于复现
        }
        torch.save(checkpoint, config.MODEL_SAVE_PATH)
        print(f"Model saved to {config.MODEL_SAVE_PATH}")