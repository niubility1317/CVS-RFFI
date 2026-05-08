import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import datetime

# 导入自定义模块
from dataset import WiFiRFFIDataset
from model import CVSincNet
from DataAugmentation import RFFIAugmentor  # <--- 新增导入

# ================= 配置区域 =================
DATASET_DIR = "./Dataset_ORALCE"  
LOG_DIR = "./log"
WEIGHT_DIR = "./weight"

BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 增强配置
USE_AUGMENTATION = True
NOISE_LEVEL = 0.005  # 噪声强度，如果数据很干净可以稍微调大到 0.01
# ===========================================

def get_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")
    
    def log_func(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")     
    return log_func

def main():
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    
    logger = get_logger(LOG_DIR)
    logger(f"Starting Training at {datetime.datetime.now()}")
    logger(f"Device: {DEVICE}")
    logger(f"Data Augmentation Enabled: {USE_AUGMENTATION}")

    # 1. 加载数据
    try:
        train_ds = WiFiRFFIDataset(DATASET_DIR, mode='train')
        test_ds = WiFiRFFIDataset(DATASET_DIR, mode='test')
    except Exception as e:
        logger(f"Error loading datasets: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 初始化增强器 (仅用于训练)
    augmentor = RFFIAugmentor(
        use_phase_rotate=True,
        use_noise=True,
        use_amp_scale=True,
        noise_std=NOISE_LEVEL
    )

    # 3. 初始化模型
    model = CVSincNet(num_classes=16).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_acc = 0.0

    # 4. 训练循环
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # [关键步骤] 应用数据增强
            # 这一步在 GPU 上进行，速度非常快
            if USE_AUGMENTATION:
                with torch.no_grad(): # 增强过程不需要计算梯度
                    inputs = augmentor(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # --- Validation (不使用增强) ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                # 注意：测试时千万不要 augmentor(inputs)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        # 日志记录
        log_msg = (f"Epoch [{epoch+1}/{EPOCHS}] "
                   f"Time: {epoch_time:.1f}s | "
                   f"Loss: {avg_train_loss:.4f} | "
                   f"Train Acc: {train_acc:.2f}% | "
                   f"Test Acc: {test_acc:.2f}%")
        logger(log_msg)

        # 保存模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "best_model_Aug.pth"))
            logger(f"  >>> Best Model Saved! (Acc: {best_acc:.2f}%)")
        
        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "last_model_Aug.pth"))

if __name__ == "__main__":
    main()