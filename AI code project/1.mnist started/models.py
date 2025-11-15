# models.py

import torch
import torch.nn as nn
import config  # 导入配置

class SimpleMNISTModel(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, activation1, activation2):
        super(SimpleMNISTModel, self).__init__()
        
        # 根据字符串配置动态选择激活函数
        act_funcs = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01)
            # 可以轻松添加更多激活函数
        }
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_size1)
        self.act1 = act_funcs[activation1]
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.act2 = act_funcs[activation2]
        self.fc3 = nn.Linear(hidden_size2, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

def create_model():
    """
    根据config.py中的配置，创建并返回模型实例。
    """
    model_params = config.MODEL_CONFIG
    model = SimpleMNISTModel(
        hidden_size1=model_params['hidden_size1'],
        hidden_size2=model_params['hidden_size2'],
        activation1=model_params['activation1'],
        activation2=model_params['activation2']
    ).to(config.DEVICE)
    
    print(f"Model '{model_params['type']}' created successfully.")
    # print(model) # 如果想查看模型结构，可以取消注释
    return model