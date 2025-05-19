# 用来训练一个经典的CNN模型
# 以便将其性能与 FR_Qcnn.py 中实现的量子CNN模型进行比较
# 将这个TensorFlow全部改为pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime
import sys
import os

# 忽略警告 (与FR.py一致)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# --- 1. 数据集类定义 ---
class FaceDatasetCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        # 图像数据是前100*100列
        self.images = self.df.iloc[:, :100*100].values.astype(np.float32)
        # 标签是最后一列
        self.labels = self.df.iloc[:, -1].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(100, 100, 1) # Reshape to (H, W, C)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        # PyTorch CNN期望输入是 (C, H, W)
        # 如果transform中没有 ToTensor()，则需要手动转换并调整维度
        # 如果transform包含 ToTensor()，它会自动处理归一化到[0,1]和维度转换
        if not isinstance(image, torch.Tensor): # 确保是Tensor
            image = transforms.ToTensor()(image) # 这会将(H,W,C)变为(C,H,W)并归一化到[0,1]
        elif image.shape[0] != 1 and image.shape[2] == 1: # 如果是(H,W,C)的Tensor
             image = image.permute(2, 0, 1)


        return image, label

# --- 2. 定义PyTorch CNN模型 ---
# 模型结构尽量与FR.py中的Keras模型保持一致
class FaceCNN(nn.Module):
    def __init__(self, num_classes=5): # FR.py中最后Dense层是5个输出单元
        super(FaceCNN, self).__init__()
        self.model_name = 'Face_trained_model_pytorch_'+datetime.now().strftime("%H_%M_%S_")

        # Keras: model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(100, 100, 1)))
        # PyTorch: nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # Keras的 'relu' 激活通常在层定义之后作为单独的激活层或在forward中调用
        # Keras的 input_shape 在第一层指定，PyTorch的输入形状在forward时体现

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # padding=1使3x3卷积后尺寸不变
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2), # padding=2使5x5卷积后尺寸不变
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (100,100) -> (50,50)
            nn.Dropout(0.2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (50,50) -> (25,25)
            nn.Dropout(0.2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (25,25) -> (12,12) (如果25/2，向下取整)
                                                    # Keras的MaxPooling默认向下取整，PyTorch也是
                                                    # 25x25 -> MaxPool2d(2) -> 12x12
            nn.Dropout(0.2)
        )

        # Flatten操作在forward中进行
        # 计算Flatten之后的维度: 256通道 * 12 * 12 (基于输入100x100，三次2x2池化)
        # 100 -> 50 -> 25 -> 12
        self.fc_block = nn.Sequential(
            nn.Linear(256 * 12 * 12, 256),
            nn.BatchNorm1d(256), # 注意是BatchNorm1d因为是全连接层
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # 输出层，PyTorch的CrossEntropyLoss内置了softmax
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1) # Flatten. x.size(0)是batch_size
        x = self.fc_block(x)
        return x

# --- 3. 数据加载和预处理 ---
# 定义数据变换 (FR.py中的ImageDataGenerator的rescale等效于ToTensor)
# FR.py中的数据增强比较复杂，这里先用基础的ToTensor和Normalize
# 如果要完全对应FR.py中的增强，需要使用transforms.RandomRotation, RandomAffine等
train_transforms = transforms.Compose([
    # ToTensor会把PIL Image或者numpy.ndarray (H x W x C) 从范围 [0, 255] 转换成 torch.FloatTensor (C x H x W) 范围 [0.0, 1.0]
    # 我们的数据已经是numpy (H,W,C) 且值在0-1之间(如果CSV里是0-255，ToTensor会处理；如果是0-1，它也接受)
    # FaceDatasetCSV中已经reshape成(H,W,C)并转为float32
    # transforms.ToTensor(), # FaceDatasetCSV的__getitem__中已经处理
    transforms.Normalize(mean=[0.5], std=[0.5]) # 假设是单通道灰度图，均值0.5，标准差0.5
                                                # 这个需要根据你的数据实际分布来定，或者像Keras一样只做rescale (ToTensor已做)
])

val_transforms = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据 (假设test.csv就是我们的总数据集)
# 注意：FR.py 中是从 test.csv 加载，然后自己划分 train/test。这里我们也这样做。
# 如果你的 test.csv 仅仅是测试集，你需要一个包含训练数据和标签的 train.csv
csv_file_path = 'test.csv' # 请确保这个文件存在且格式正确
if not os.path.exists(csv_file_path):
    print(f"错误: CSV文件 '{csv_file_path}' 不存在。请提供正确的数据文件路径。")
    sys.exit()

full_dataset_df = pd.read_csv(csv_file_path, index_col=0)
X_all = full_dataset_df.iloc[:, :100*100].values.astype(np.float32)
y_all = full_dataset_df.iloc[:, -1].values.astype(np.int64)

# 划分训练集和验证集
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_all, y_all, random_state=42, test_size=0.15, stratify=y_all)


# 自定义Dataset类用于NumPy数组
class NumpyImageDataset(Dataset):
    def __init__(self, images_np, labels_np, transform=None):
        self.images = images_np
        self.labels = labels_np
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(100, 100, 1) # (H, W, C)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # 确保输出是 (C, H, W) 的Tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image) # 归一化到[0,1]并转为 (C,H,W)
        elif image.shape[0] != 1 and image.shape[2] == 1: # 如果是(H,W,C)的Tensor
             image = image.permute(2, 0, 1)
        
        return image, label

train_dataset = NumpyImageDataset(X_train_np, y_train_np, transform=train_transforms)
val_dataset = NumpyImageDataset(X_val_np, y_val_np, transform=val_transforms)


batch_size = 256 # 与FR.py一致
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# --- 4. 模型实例化、损失函数、优化器 ---
# num_classes 需要根据你的数据集确定。FR.py中最后是Dense(5, ...), 但数据加载时用了 num_classes=4，
# 并且to_categorical时 num_classes = 1 + df.loc[:, 'class'].unique().shape[0]
# 这里我们假设 num_classes 与FR.py中的Dense(5,...)一致
num_actual_classes = len(np.unique(y_all))
print(f"数据集中实际的类别数量: {num_actual_classes}")
# 通常 num_classes 参数应该等于这个值。FR.py 中 `to_categorical` 的 `num_classes` 参数可能需要检查。
# 如果 y_all 的标签是从0开始的，例如0,1,2,3,4，那么num_actual_classes就是5。
# 我们这里直接使用FR.py模型定义中的5
model_pytorch = FaceCNN(num_classes=5)


criterion = nn.CrossEntropyLoss() # 包含了Softmax
optimizer = optim.RMSprop(model_pytorch.parameters(), lr=0.001) # 与FR.py一致
# 学习率调度器 (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True) # FR.py的patience是200，这里改小一点以便更快看到效果

# --- 5. 训练循环 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model_pytorch.to(device)

epochs = 50 # 与FR.py一致 (FR.py的patience=200，epochs=50可能跑不完patience)
best_val_accuracy = 0.0
patience_counter = 0 # 用于早停
early_stopping_patience = 10 # FR.py的patience是200，这里改小一点

# 创建保存模型的文件夹
if not os.path.exists('models_pytorch'):
    os.makedirs('models_pytorch')

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model_pytorch.train() # 设置为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_pytorch(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # --- 验证 ---
    model_pytorch.eval() # 设置为评估模式
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # 评估时不需要计算梯度
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_pytorch(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    epoch_val_acc = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    scheduler.step(epoch_val_loss) # 更新学习率

    # 模型保存 (类似ModelCheckpoint)
    if epoch_val_acc > best_val_accuracy:
        best_val_accuracy = epoch_val_acc
        torch.save(model_pytorch.state_dict(), os.path.join('models_pytorch', model_pytorch.model_name + '.pth'))
        print(f"模型已保存，最佳验证准确率: {best_val_accuracy:.4f}")
        patience_counter = 0 # 重置早停计数器
    else:
        patience_counter += 1

    # 早停 (类似EarlyStopping)
    if patience_counter >= early_stopping_patience:
        print("早停触发！")
        break

# --- 6. 评估和可视化 (与FR.py类似) ---
# 加载最佳模型进行最终评估
if os.path.exists(os.path.join('models_pytorch', model_pytorch.model_name + '.pth')):
    model_pytorch.load_state_dict(torch.load(os.path.join('models_pytorch', model_pytorch.model_name + '.pth')))
    print("已加载最佳模型进行最终评估。")

model_pytorch.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader: # 使用验证集进行最终评估
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_pytorch(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_accuracy = accuracy_score(all_labels, all_preds)
print(f"\n最终验证集准确率 (使用最佳模型): {final_accuracy*100:.2f}%")
print("\n分类报告:")
# 注意：target_names 需要根据你的类别实际名称来设置
# 如果类别是0,1,2,3,4，可以这样：
unique_labels = sorted(np.unique(y_all))
target_names_str = [str(label) for label in unique_labels]
print(classification_report(all_labels, all_preds, target_names=target_names_str if len(target_names_str) == num_actual_classes else None))

print("\n混淆矩阵:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)
# 可视化混淆矩阵 (可选，需要seaborn)
# import seaborn as sns
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_str, yticklabels=target_names_str)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()


# 绘制训练过程中的准确率和损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

print("PyTorch版本CNN训练和评估完成。")