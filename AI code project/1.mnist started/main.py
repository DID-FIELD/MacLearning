# main.py
import data_loader
import models
import trainer
import utils
import config

def main():
    print("Starting MNIST Training...")
    print(f"Using device: {config.DEVICE}")
    
    # 1. 加载数据
    train_loader, test_loader = data_loader.get_data_loaders()
    
    # 2. 创建模型
    model = models.create_model()
    
    # 3. 开始训练
    results = trainer.train(model, train_loader, test_loader)
    
    # 4. 处理并保存结果
    utils.save_results(results)
    utils.plot_and_save_results(results)
    
    print("\nTraining process completed successfully!")

if __name__ == '__main__':
    main()