# cnn_cifar10.py - RTX 1660 Super向け最適化版
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# RTX 1660 Super最適化設定
BATCH_SIZE = 32  # 6GB VRAM向け最適化
EPOCHS = 10
LEARNING_RATE = 0.001

# デバイス設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# RTX 1660 Super向け最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 固定入力サイズに最適化
    torch.cuda.set_per_process_memory_fraction(0.9)  # メモリ使用量制限

# データ前処理
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# データセットの準備
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=4, pin_memory=True  # GPU転送最適化
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=4, pin_memory=True
)

# RTX 1660 Super向け軽量CNNモデル
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(OptimizedCNN, self).__init__()
        
        # 最適化された畳み込み層
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 全結合層
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# モデルの初期化
model = OptimizedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# メモリ使用量監視関数
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory - 割り当て: {allocated:.2f}GB, 予約: {reserved:.2f}GB")

# 訓練関数
def train_model():
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            monitor_memory()
    
    return total_loss / len(trainloader)

# テスト関数
def test_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# 訓練実行
print("=== RTX 1660 Super CNN Training ===")
print(f"バッチサイズ: {BATCH_SIZE}")
print(f"エポック数: {EPOCHS}")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    start_time = time.time()
    train_loss = train_model()
    epoch_time = time.time() - start_time
    
    accuracy = test_model()
    
    print(f"エポック {epoch+1} - 損失: {train_loss:.4f}, 精度: {accuracy:.2f}%, 時間: {epoch_time:.2f}秒")
    
    # メモリクリーンアップ
    torch.cuda.empty_cache()

print("\n訓練完了!")
