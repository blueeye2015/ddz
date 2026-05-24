#!/usr/bin/env python3
"""
MobileNetV3-Small 分类器训练
输入: dataset/by_class/ 下的 15 类裁剪图
输出: card_classifier_best.pth
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def main():
    # 检查 PyTorch
    try:
        import torch
    except ImportError:
        print("ERR 未安装 PyTorch，请先运行:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  使用设备: {device}")
    
    # ========== 配置 ==========
    data_dir = 'dataset/by_class'
    num_epochs = 50
    batch_size = 32
    lr = 0.001
    
    if not os.path.exists(data_dir):
        print(f"ERR 数据目录不存在: {data_dir}")
        print("   请确保 dataset/by_class/ 下有 15 个类别的子文件夹")
        sys.exit(1)
    
    # ========== 数据增强 ==========
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ========== 数据集 ==========
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)
    
    # 按 8:2 划分
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 验证集使用无增强的 transform
    val_set.dataset.transform = val_transform
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)
    
    print(f"Data 类别 ({num_classes}): {full_dataset.classes}")
    print(f"Data 训练: {len(train_set)}, 验证: {len(val_set)}")
    
    # ========== 模型 ==========
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # ========== 训练循环 ==========
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # ---- 验证 ----
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f}  "
              f"Val Loss: {val_loss/len(val_loader):.4f}  "
              f"Val Acc: {val_acc:5.2f}%  "
              f"Time: {elapsed:.0f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'card_classifier_best.pth')
            print(f"  OK 保存最佳模型 (acc={val_acc:.2f}%)")
    
    print(f"\nDone 训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"Saved 模型已保存: card_classifier_best.pth")
    print(f"Map 类别映射: {dict(enumerate(full_dataset.classes))}")


if __name__ == "__main__":
    main()
