# gpu_test.py - GPU環境検証スクリプト

## 概要

`gpu_test.py`は、RTX 1660 SuperのGPU環境が正しく設定されているかを包括的に検証するスクリプトです。PyTorchとCUDAの動作確認から、実際のGPU計算テストまでを自動化して実行します。

## 主要機能

### 1. PyTorch基本情報の確認
- PyTorchバージョンの表示
- Pythonバージョンの表示
- システム環境の基本チェック

### 2. CUDA環境の検証
- CUDAの利用可能性確認
- CUDAバージョンの表示
- cuDNNの有効性とバージョン確認

### 3. GPU ハードウェア情報の取得
- 利用可能なGPU数の確認
- 各GPUの詳細プロパティ表示：
  - GPUモデル名
  - VRAMメモリ容量
  - 計算能力 (Compute Capability)
  - SMプロセッサ数

### 4. RTX 1660 Super 特有の検証
- RTX 1660 Superの正確な認識確認
- Tensor Coreの有無確認（RTX 1660 SuperはTensor Core非搭載のため警告表示）
- 混合精度訓練の制限事項の通知

### 5. 実際のGPU計算テスト
- GPU上での行列演算実行
- メモリ使用量の監視
- 計算パフォーマンスの測定
- メモリクリーンアップの確認

## 技術的詳細

### メモリ監視機能
```python
allocated_before = torch.cuda.memory_allocated(0) / 1024**3
# 計算実行
allocated_after = torch.cuda.memory_allocated(0) / 1024**3
```
計算前後のVRAM使用量をGBスケールで監視し、メモリリークの検出を行います。

### 行列乗算ベンチマーク
1000x1000の行列同士の乗算を実行し、実行時間を測定：
- GPU上でのテンソル作成
- 行列乗算の実行時間測定
- GPU同期による正確な時間計測

### エラーハンドリング
- CUDA利用不可時の適切なエラーメッセージ表示
- GPU計算失敗時の例外キャッチと詳細表示
- 戻り値による成功/失敗の判定

## 使用方法

```bash
uv run python src/gpu_test.py
```

## 出力例

```
=== RTX 1660 Super GPU Setup Verification ===
PyTorch version: 2.7.0+cu118
Python version: 3.11.x
CUDA available: True
CUDA version: 11.8
cuDNN enabled: True
GPU count: 1
GPU 0: NVIDIA GeForce RTX 1660 SUPER
  メモリ: 6.0 GB
  計算能力: 7.5
  SM数: 22
✅ RTX 1660 Super が正常に認識されています
⚠️  注意: RTX 1660 SuperはTensor Coreを搭載していません

=== GPU計算テスト ===
割り当てメモリ: 0.01 GB
計算後メモリ: 0.01 GB
行列乗算時間: 0.0023 秒
結果形状: torch.Size([1000, 1000])
結果デバイス: cuda:0
✅ GPU計算テスト: 成功
```

## RTX 1660 Super最適化ポイント

1. **VRAMメモリ制限**: 6GBのVRAM制限を考慮した小さなテストサイズ
2. **Tensor Core非対応**: 混合精度訓練の制限について明確な警告
3. **メモリクリーンアップ**: テスト後の適切なメモリ解放
4. **計算能力確認**: SM数とcompute capabilityによる性能把握

## 関連ファイル

- `pyproject.toml`: PyTorch CUDA 11.8の依存関係設定
- `CLAUDE.md`: 実行コマンドと環境設定の詳細