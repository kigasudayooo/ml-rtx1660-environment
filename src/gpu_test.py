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
