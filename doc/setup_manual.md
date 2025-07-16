# PyTorch GPU環境構築完全マニュアル：RTX 1660 Super on Debian

RTX 1660 Superを搭載したLinux Debian環境で、最適化されたPyTorch機械学習環境を構築する完全ガイドです。**RTX 1660 Super（6GB VRAM、Turing TU116、計算能力7.5）に最適化されたCUDA 11.8とPyTorch 2.7の組み合わせ**を使用し、初学者でも確実に環境構築できるよう詳細に解説します。

## 1. 初期設定とハードウェア準備

### システム要件の確認

RTX 1660 Superは**Turing アーキテクチャ**を採用し、6GB GDDR6メモリ、計算能力7.5を持つミドルレンジGPUです。重要な点として、**Tensor Coreを搭載していない**ため、混合精度訓練の恩恵は限定的です。

```bash
# システム情報の確認
uname -a
lsb_release -a
lspci | grep -i nvidia

# 推奨システム要件
# - OS: Debian 12 "Bookworm"
# - RAM: 最低8GB（推奨16GB）
# - 電源: 最低500W（8ピンPCIeコネクタ必須）
# - 空き容量: 5-10GB
```

### NVIDIAドライバのインストール（Debian公式推奨方式）

**Debian公式リポジトリを使用する理由：**
- システムとの完全統合で安定性最高
- 依存関係の自動解決
- 簡単なアップデートとトラブル解決
- RTX 1660 Superに最適な535系ドライバが利用可能

#### **ステップ1: システム準備**

```bash
# システムパッケージの更新
sudo apt update && sudo apt upgrade -y

# 必要なパッケージのインストール
sudo apt install -y build-essential dkms linux-headers-$(uname -r)

# non-freeリポジトリの追加（プロプライエタリドライバ用）
echo "deb http://deb.debian.org/debian/ bookworm main contrib non-free non-free-firmware" | sudo tee /etc/apt/sources.list.d/non-free.list
sudo apt update
```

#### **ステップ2: nouveauドライバの完全無効化**

**重要：** nouveauドライバとNVIDIAドライバは同時に使用できません。

```bash
# ブラックリストファイルの作成
sudo tee /etc/modprobe.d/blacklist-nouveau.conf <<EOF
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOF

# GRUBブートパラメータでも無効化
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nouveau.modeset=0"/' /etc/default/grub

# 設定を適用
sudo update-initramfs -u
sudo update-grub
```

#### **ステップ3: セキュアブートの確認と対処**

**重要：** セキュアブートが有効だとNVIDIAドライバがロードできません。

```bash
# セキュアブートの状態確認
mokutil --sb-state

# "SecureBoot enabled" と表示された場合は無効化が必要
```

**セキュアブート無効化手順：**
1. PCを再起動してBIOS/UEFI設定に入る（F2、F12、Delete等）
2. "Security" → "Secure Boot" または "Boot" → "Secure Boot Control" を探す
3. "Secure Boot" を **Disabled** に変更
4. 設定を保存して再起動

#### **ステップ4: NVIDIAドライバのインストール**

```bash
# 既存のNVIDIAドライバがあれば削除
sudo apt purge "*nvidia*" "*cuda*" "libcudnn*" -y

# GPUの検出（オプション）
sudo apt install nvidia-detect -y
nvidia-detect

# Debian公式NVIDIAドライバのインストール
sudo apt install nvidia-driver firmware-misc-nonfree -y

# 再起動（必須）
sudo reboot
```

### インストール確認と初期設定

#### **基本的な動作確認**

```bash
# ドライバの動作確認
nvidia-smi

# 期待される出力例:
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.247.01     Driver Version: 535.247.01   CUDA Version: 12.2          |
# +---------------------------------------------------------------------------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce GTX 1660 SUPER   Off | 00000000:01:00.0  On |                  N/A |
# | 42%   35C    P8              8W / 125W |   165MiB /  6144MiB |      0%      Default |
# |                                         |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
```

#### **トラブルシューティング: nvidia-smiが失敗する場合**

**エラー**: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"

**診断手順:**
```bash
# 1. カーネルモジュールの確認
lsmod | grep nvidia

# 2. nouveauドライバの確認
lsmod | grep nouveau

# 3. インストール済みパッケージの確認
dpkg -l | grep nvidia

# 4. DKMSの状況確認
sudo dkms status
```

**解決方法:**

**ケース1: カーネルモジュールがロードされていない**
```bash
# 手動でカーネルモジュールをロード
sudo modprobe nvidia
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# ロード確認
lsmod | grep nvidia

# 動作確認
nvidia-smi
```

**ケース2: セキュアブートが有効（"Key was rejected by service"エラー）**
```bash
# セキュアブートの確認
mokutil --sb-state

# "SecureBoot enabled"の場合はBIOSで無効化が必要
# 手順: 再起動 → BIOS設定 → Security → Secure Boot → Disabled
```

**ケース3: nouveauドライバが残っている**
```bash
# nouveauの強制削除
sudo rmmod nouveau

# ブラックリスト再確認
cat /etc/modprobe.d/blacklist-nouveau.conf

# 再起動
sudo reboot
```

#### **自動起動設定**

nvidia-smiが手動で動作した場合、再起動後も自動で動くよう設定：

```bash
# カーネルモジュールの自動ロード設定
echo 'nvidia' | sudo tee -a /etc/modules
echo 'nvidia_drm' | sudo tee -a /etc/modules
echo 'nvidia_uvm' | sudo tee -a /etc/modules

# modprobe設定
sudo tee /etc/modprobe.d/nvidia.conf <<EOF
options nvidia_drm modeset=1
EOF

# systemdサービス有効化
sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced

# initramfs更新
sudo update-initramfs -u

# 再起動して確認
sudo reboot
```

## 2. CUDA ToolkitとcuDNNのインストール

### CUDA 11.8の最適化インストール

RTX 1660 SuperにはCUDA 11.8が最適です。これは**Turingアーキテクチャ向けに最適化されたヒューリスティック**を持つためです。

```bash
# CUDA Toolkitのインストール
sudo apt install nvidia-cuda-dev nvidia-cuda-toolkit -y

# インストール確認
nvcc --version
# 期待される出力: Cuda compilation tools, release 11.8, V11.8.89

# 環境変数の設定
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### cuDNNの設定

```bash
# NVIDIAリポジトリの追加
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# cuDNN 9.10.1のインストール
sudo apt install libcudnn9-dev libcudnn9-cuda-11 -y

# インストール確認
ldconfig -p | grep cudnn
```

### 動作確認テスト

```bash
# CUDAのサンプルテスト
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# 期待される出力:
# Device 0: "NVIDIA GeForce GTX 1660 SUPER"
# CUDA Capability Major/Minor version number: 7.5
# Total amount of global memory: 6144 MBytes
# Result = PASS
```

## 3. Python環境構築とuvセットアップ

### uvパッケージマネージャーのインストール

uvは従来のpipより**10-100倍高速**で、依存関係の解決が確実です。

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# シェルを再起動するか以下を実行
source ~/.bashrc

# バージョン確認
uv --version
```

### Python 3.11環境の構築

```bash
# 機械学習プロジェクトの初期化
uv init ml-rtx1660-environment
cd ml-rtx1660-environment

# Python 3.11のインストールと設定
uv python install 3.11
uv python use 3.11

# 仮想環境の確認
uv python list
```

### プロジェクト設定ファイルの作成

`pyproject.toml`を以下の内容で作成します：

```toml
[project]
name = "ml-rtx1660-environment"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.7.0",
    "torchvision>=0.22.0",
    "torchaudio>=2.7.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.3.0",
    "jupyter>=1.0.0",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cu118" }]
torchvision = [{ index = "pytorch-cu118" }]
torchaudio = [{ index = "pytorch-cu118" }]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true
```

### PyTorchとライブラリのインストール

```bash
# PyTorchのGPU版インストール（RTX 1660 Super最適化）
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu118

# 機械学習ライブラリの追加
uv add numpy pandas matplotlib seaborn scikit-learn jupyter

# 開発ツールの追加
uv add --dev pytest black ruff mypy ipykernel

# 追加ライブラリ
uv add scipy plotly openpyxl sqlalchemy

# インストール確認
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 4. 動作確認とGPU認識テスト

### 基本的なGPU認識確認

```python
# gpu_test.py
import torch
import sys

def comprehensive_gpu_test():
    print("=== RTX 1660 Super GPU Setup Verification ===")
    
    # 1. PyTorch基本情報
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # 2. CUDA利用可能性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA が利用できません。インストールを確認してください。")
        return False
    
    # 3. CUDA詳細情報
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # 4. GPU情報
    device_count = torch.cuda.device_count()
    print(f"GPU count: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  メモリ: {props.total_memory / 1024**3:.1f} GB")
        print(f"  計算能力: {props.major}.{props.minor}")
        print(f"  SM数: {props.multi_processor_count}")
    
    # 5. RTX 1660 Super特有の確認
    if "1660 SUPER" in torch.cuda.get_device_name(0):
        print("✅ RTX 1660 Super が正常に認識されています")
        
        # Tensor Coreの確認
        if props.major >= 7 and props.minor >= 5:
            print("⚠️  注意: RTX 1660 SuperはTensor Coreを搭載していません")
            print("   混合精度訓練の恩恵は限定的です")
    
    # 6. GPU計算テスト
    try:
        device = torch.device("cuda:0")
        
        # メモリ使用量テスト
        print(f"\n=== GPU計算テスト ===")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # メモリ使用量の確認
        allocated_before = torch.cuda.memory_allocated(0) / 1024**3
        print(f"割り当てメモリ: {allocated_before:.2f} GB")
        
        # 行列乗算テスト
        import time
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end_time = time.time()
        
        allocated_after = torch.cuda.memory_allocated(0) / 1024**3
        print(f"計算後メモリ: {allocated_after:.2f} GB")
        print(f"行列乗算時間: {end_time - start_time:.4f} 秒")
        print(f"結果形状: {z.shape}")
        print(f"結果デバイス: {z.device}")
        
        # メモリクリーンアップ
        del x, y, z
        torch.cuda.empty_cache()
        
        print("✅ GPU計算テスト: 成功")
        return True
        
    except Exception as e:
        print(f"❌ GPU計算テスト: 失敗 - {e}")
        return False

if __name__ == "__main__":
    comprehensive_gpu_test()
```

### 実行とテスト

```bash
# テストの実行
uv run python gpu_test.py

# 期待される出力例:
# === RTX 1660 Super GPU Setup Verification ===
# PyTorch version: 2.7.0
# CUDA available: True
# CUDA version: 11.8
# GPU 0: NVIDIA GeForce GTX 1660 SUPER
#   メモリ: 6.0 GB
#   計算能力: 7.5
# ✅ RTX 1660 Super が正常に認識されています
# ✅ GPU計算テスト: 成功
```

## 5. 実践的なサンプルコード

### CNN画像認識サンプル（RTX 1660 Super最適化）

```python
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
```

### Transformer自然言語処理サンプル

```python
# transformer_text_classification.py - RTX 1660 Super最適化版
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

# RTX 1660 Super向け設定
BATCH_SIZE = 8  # 小さなバッチサイズで6GB VRAM対応
MAX_LENGTH = 128  # 短めの系列長
EPOCHS = 3
LEARNING_RATE = 2e-5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# メモリ最適化設定
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.85)

# サンプルデータセット
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# RTX 1660 Super向け軽量分類モデル
class TextClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(TextClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 一部の層を固定してメモリ使用量を削減
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.bert.encoder.layer[:6]:  # 最初の6層を固定
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# メモリ効率的な訓練関数
def train_with_accumulation(model, dataloader, optimizer, criterion, accumulation_steps=4):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 勾配累積を使用してメモリ使用量を削減
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels) / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        if batch_idx % 10 == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, GPU Memory: {allocated:.2f}GB')
    
    return total_loss / len(dataloader)

# サンプルデータの準備
sample_texts = [
    "This movie is amazing!",
    "I hate this film.",
    "Great story and acting.",
    "Boring and predictable.",
    "Love the cinematography.",
    "Not worth watching.",
] * 100  # データを拡張

sample_labels = [1, 0, 1, 0, 1, 0] * 100  # 1: positive, 0: negative

# モデルとトークナイザーの初期化
model_name = "distilbert-base-uncased"  # 軽量版BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TextClassifier(model_name, num_classes=2).to(device)

# データセットとデータローダー
dataset = TextDataset(sample_texts, sample_labels, tokenizer, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 最適化とロス関数
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 訓練実行
print("=== RTX 1660 Super Transformer Training ===")
print(f"モデル: {model_name}")
print(f"バッチサイズ: {BATCH_SIZE}")
print(f"最大系列長: {MAX_LENGTH}")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    start_time = time.time()
    train_loss = train_with_accumulation(model, dataloader, optimizer, criterion)
    epoch_time = time.time() - start_time
    
    print(f"エポック {epoch+1} - 損失: {train_loss:.4f}, 時間: {epoch_time:.2f}秒")
    
    # メモリクリーンアップ
    torch.cuda.empty_cache()

print("\n訓練完了!")
```

### テーブルデータ機械学習サンプル

```python
# tabular_ml.py - RTX 1660 Super向けテーブルデータ学習
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# RTX 1660 Super向け設定
BATCH_SIZE = 256  # テーブルデータなので大きなバッチサイズ
EPOCHS = 100
LEARNING_RATE = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# サンプルデータセットの生成
X, y = make_classification(
    n_samples=10000,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    n_classes=3,
    random_state=42
)

# データの前処理
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# スケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PyTorchテンソルに変換
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# データローダーの作成
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=True
)

# GPU効率的な深層ニューラルネットワーク
class TabularNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(TabularNet, self).__init__()
        
        layers = []
        
        # 入力層
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # 隠れ層
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # 出力層
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# モデルの初期化
model = TabularNet(
    input_size=50,
    hidden_sizes=[512, 256, 128, 64],
    num_classes=3,
    dropout_rate=0.3
).to(device)

# 最適化とロス関数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 訓練関数
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total

# テスト関数
def test_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total

# 訓練実行
print("=== RTX 1660 Super Tabular ML Training ===")
print(f"データサイズ: {X_train.shape}")
print(f"バッチサイズ: {BATCH_SIZE}")

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = test_epoch(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    scheduler.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1:3d}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# 結果の可視化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

print(f"\n最終テスト精度: {test_accuracies[-1]:.2f}%")
```

## 6. トラブルシューティング

### よくあるエラーとその解決方法

**エラー1: "CUDA not available"**
```bash
# 診断コマンド
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# 解決方法
# 1. ドライバの再インストール
sudo apt purge "*nvidia*" -y
sudo apt install nvidia-driver -y
sudo reboot

# 2. PyTorchの再インストール
pip uninstall torch torchvision torchaudio
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu118
```

**エラー2: OutOfMemoryError**
```python
# RTX 1660 Super向けメモリ最適化
torch.cuda.set_per_process_memory_fraction(0.8)  # 80%に制限
torch.cuda.empty_cache()  # メモリクリーンアップ

# 勾配累積を使用
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**エラー3: FP16でのNaN値**
```python
# RTX 1660 SuperではFP16を避ける
# tensor coreがないため恩恵が少なく、CUDA 11.1で既知の問題
model = model.float()  # FP32を使用
```

### パフォーマンスチューニングのポイント

**最適なバッチサイズの発見**
```python
def find_optimal_batch_size(model, sample_input):
    """RTX 1660 Super向け最適バッチサイズ探索"""
    batch_sizes = [8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        try:
            # メモリクリーンアップ
            torch.cuda.empty_cache()
            
            # バッチサイズでテスト
            batch_input = sample_input.repeat(batch_size, 1, 1, 1)
            output = model(batch_input)
            
            # メモリ使用量確認
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"バッチサイズ {batch_size}: メモリ使用量 {memory_used:.2f}GB")
            
            if memory_used > 5.5:  # 6GB中5.5GB以上で警告
                print(f"⚠️  バッチサイズ {batch_size} はメモリ使用量が多すぎます")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ バッチサイズ {batch_size}: メモリ不足")
                break
    
    return batch_size // 2  # 安全マージンを取る
```

**GPU使用率の監視**
```python
import GPUtil
import threading
import time

def monitor_gpu():
    """GPU使用率の継続監視"""
    while True:
        GPUs = GPUtil.getGPUs()
        for gpu in GPUs:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  使用率: {gpu.load*100:.1f}%")
            print(f"  メモリ: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            print(f"  温度: {gpu.temperature}°C")
        time.sleep(1)

# 監視スレッドの開始
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()
```

### RTX 1660 Super特有の最適化

**メモリ効率的な設定**
```python
# RTX 1660 Super向け最適化設定
torch.backends.cudnn.benchmark = True  # 固定サイズに最適化
torch.backends.cudnn.enabled = True
torch.cuda.set_per_process_memory_fraction(0.9)  # 90%使用許可

# データローダー最適化
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,  # GPU転送高速化
    persistent_workers=True  # ワーカー再利用
)
```

**温度管理**
```bash
# GPU温度の監視
nvidia-smi -q -d TEMPERATURE

# ファン速度の調整（必要に応じて）
nvidia-settings -a [gpu:0]/GPUFanControlState=1
nvidia-settings -a [fan:0]/GPUTargetFanSpeed=60
```

## 📋 実際のトラブルシューティング経験から

### 🔍 **一般的な問題の診断フロー**

**Step 1: 基本確認**
```bash
# GPU認識確認
lspci | grep -i nvidia

# ドライバ状況確認
nvidia-smi
lsmod | grep nvidia

# セキュアブート確認
mokutil --sb-state
```

**Step 2: 段階的解決**
1. **セキュアブート無効化** (最も多い原因)
2. **nouveauドライバ完全除去**
3. **カーネルモジュール手動ロード**
4. **自動起動設定**

### ✅ **成功パターンの確認方法**

**正常な状態:**
```bash
# nvidia-smiの期待出力
$ nvidia-smi
Wed Jul 16 12:42:06 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.247.01             Driver Version: 535.247.01   CUDA Version: 12.2   |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1660 ...    On  | 00000000:01:00.0 Off |                  N/A |
| 41%   38C    P8              11W / 125W |    165MiB /  6144MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+

# カーネルモジュールの確認
$ lsmod | grep nvidia
nvidia_uvm           1560576  0
nvidia_drm             69632  1
nvidia_modeset       1142784  1 nvidia_drm
nvidia              56401920  2 nvidia_uvm,nvidia_modeset
```

### 🚨 **最重要ポイント**

1. **セキュアブート**: RTX 1660 Superで機械学習を行う場合、セキュアブートは無効化推奨
2. **Debian公式ドライバ**: システムとの統合性が高く、トラブル解決が容易
3. **段階的アプローチ**: 問題は一つずつ解決し、各段階で動作確認を行う

## 📋 Debian公式ドライバのメリット・デメリット

### ✅ **Debian公式の利点**
- **システム統合**: パッケージ管理システムとの完全統合
- **自動更新**: `apt upgrade`でドライバも一緒に更新
- **DKMS対応**: カーネル更新時の自動再構築
- **依存関係**: 自動解決で競合回避
- **トラブル解決**: `apt remove`で簡単に元に戻せる
- **安定性**: Debianチームによるテスト済み

### ⚠️ **制限事項**
- **更新タイミング**: 最新版の提供が少し遅れる
- **バージョン選択**: 特定バージョンの指定が難しい

### 🔄 **NVIDIA公式を選ぶべきケース**
- 最新のCUDA機能が必要
- 開発中の機能をテストする場合
- Debian公式で解決できない問題がある場合

**結論**: RTX 1660 SuperでのPyTorch機械学習には、Debian公式ドライバが最適解です。実際のトラブルシューティング経験からも、セキュアブート無効化とDebian公式ドライバの組み合わせが最も安定した動作を実現します。
