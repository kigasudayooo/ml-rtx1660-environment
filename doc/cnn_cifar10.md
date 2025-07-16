# cnn_cifar10.py - RTX 1660 Super最適化CNN実装

## 概要

`cnn_cifar10.py`は、RTX 1660 SuperのGPU環境に最適化されたCIFAR-10画像分類CNNモデルの実装です。6GB VRAMの制限を考慮し、メモリ効率的な設計とリアルタイムメモリ監視を特徴とします。

## 主要機能

### 1. RTX 1660 Super向け最適化設定
- **バッチサイズ**: 32（6GB VRAM対応）
- **メモリ使用量制限**: 90%に設定
- **cuDNN最適化**: 固定入力サイズ向けベンチマーク有効化

### 2. データ処理パイプライン
- **CIFAR-10データセット**: 自動ダウンロードと前処理
- **データ拡張**: ランダム水平フリップとクロッピング
- **正規化**: CIFAR-10標準統計値を使用
- **GPU転送最適化**: Pin memoryとnum_workers=4設定

### 3. 軽量CNN アーキテクチャ
```
入力 (3x32x32) 
→ Conv2d(64) + ReLU + MaxPool 
→ Conv2d(128) + ReLU + MaxPool 
→ Conv2d(256) + ReLU + MaxPool 
→ Dropout(0.5) + FC(512) + Dropout(0.5) + FC(10)
```

### 4. メモリ監視システム
- エポック毎のGPUメモリ使用量追跡
- バッチ処理中のリアルタイム監視
- 自動メモリクリーンアップ

## 技術的詳細

### OptimizedCNN モデル設計

#### 畳み込み層の構成
```python
self.conv1 = nn.Conv2d(3, 64, 3, padding=1)    # 32x32 → 32x32
self.conv2 = nn.Conv2d(64, 128, 3, padding=1)  # 16x16 → 16x16  
self.conv3 = nn.Conv2d(128, 256, 3, padding=1) # 8x8 → 8x8
```

各畳み込み層の後にMaxPooling(2x2)を適用し、特徴マップサイズを半減します。

#### 全結合層の設計
```python
self.fc1 = nn.Linear(256 * 4 * 4, 512)  # 4096 → 512
self.fc2 = nn.Linear(512, 10)           # 512 → 10クラス
```

DropoutとReLU活性化関数を組み合わせ、過学習を防止します。

### メモリ最適化戦略

#### GPU設定
```python
torch.backends.cudnn.benchmark = True  # 固定サイズ最適化
torch.cuda.set_per_process_memory_fraction(0.9)  # 90%制限
```

#### DataLoader最適化
```python
DataLoader(batch_size=32, num_workers=4, pin_memory=True)
```
- `pin_memory=True`: CPU-GPU転送の高速化
- `num_workers=4`: マルチプロセスデータ読み込み

### 訓練プロセス

#### 1. フォワードパス
```python
data, target = data.to(device), target.to(device)
output = model(data)
loss = criterion(output, target)
```

#### 2. バックプロパゲーション
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### 3. メモリ監視
```python
allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3
```

100バッチごとにメモリ使用量をGB単位で出力します。

### パフォーマンス特性

#### 典型的な実行時間
- **1エポック**: 約30-60秒（データセットサイズ依存）
- **メモリ使用量**: 約1.5-2.0GB（ピーク時）
- **精度**: 10エポック後約70-80%

#### RTX 1660 Super固有の制約
- **Tensor Core非対応**: 混合精度の恩恵なし
- **メモリ帯域幅**: 336 GB/s（RTX 30xx系より低い）
- **CUDA Core数**: 1408（中級クラス）

## 使用方法

```bash
uv run python src/cnn_cifar10.py
```

## 出力例

```
=== RTX 1660 Super CNN Training ===
バッチサイズ: 32
エポック数: 10

Epoch 1/10
Batch 0, Loss: 2.3026
GPU Memory - 割り当て: 1.23GB, 予約: 1.50GB
Batch 100, Loss: 1.8456
GPU Memory - 割り当て: 1.24GB, 予約: 1.50GB
エポック 1 - 損失: 1.9234, 精度: 32.45%, 時間: 45.23秒

Epoch 2/10
...
```

## ファイル構造

### データディレクトリ
```
./data/
├── cifar-10-batches-py/    # 自動ダウンロードされるCIFAR-10データ
│   ├── data_batch_1
│   ├── data_batch_2
│   └── ...
```

### 関連設定
- **学習率**: 0.001（Adam最適化）
- **損失関数**: CrossEntropyLoss
- **評価指標**: Top-1精度

## 拡張可能性

### モデル改良案
1. **ResNet blocks**: 残差接続の追加
2. **Batch Normalization**: 各畳み込み層後に追加
3. **Learning Rate Scheduler**: 動的学習率調整

### メモリ効率改善
1. **Mixed Precision**: 将来のGPU向け（RTX 1660 Superは非対応）
2. **Gradient Checkpointing**: メモリ使用量削減
3. **Model Parallelism**: 大型モデル向け分散処理

## 関連ファイル

- `gpu_test.py`: GPU環境の事前確認
- `pyproject.toml`: PyTorchとtorchvisionの依存関係
- `CLAUDE.md`: 実行コマンドと環境設定