# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # 引入进度条，让训练过程更直观
import config
import utils

def train_one_epoch(model, train_loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 更新进度条信息
        loop.set_description(f"Train Epoch")
        loop.set_postfix(loss=running_loss/(batch_idx+1), acc=100*correct/total)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

def validate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    print(f'Validation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

def train(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    scaler = torch.cuda.amp.GradScaler()

    results = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_test_acc = 0.0

    for epoch in range(config.EPOCHS):
        print(f'\nEpoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 20)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        
        test_loss, test_acc = validate(model, test_loader, criterion)

        # 记录结果
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        
        # 如果当前模型是最好的，就保存
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            utils.save_model(model, epoch, test_acc)
            
    return results