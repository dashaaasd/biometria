import os
import time
import torch
import numpy as np

from config import Config
from models import GhostNetV2_FER, ViT_FER


def benchmark_model(model, model_path, device, name="Model"):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # Параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[Параметры]")
    print(f"  Всего:     {total_params:,}")
    print(f"  Обучаемых: {trainable_params:,}")

    # Размер файла
    size_mb = None
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"\n[Размер модели]")
        print(f"  Файл .pth: {size_mb:.2f} MB")

    # Латентность на GPU
    model.eval()
    dummy = torch.randn(1, 1, 48, 48).to(device)

    with torch.no_grad():
        for _ in range(20):
            model(dummy)

    times_gpu = []
    with torch.no_grad():
        for _ in range(200):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            torch.cuda.synchronize()
            times_gpu.append(time.perf_counter() - start)

    print(f"\n[Латентность GPU (batch=1)]")
    print(f"  Среднее: {np.mean(times_gpu)*1000:.2f} ms")
    print(f"  FPS:      {1/np.mean(times_gpu):.1f}")

    # Латентность на CPU
    model_cpu = model.cpu()
    dummy_cpu = torch.randn(1, 1, 48, 48)

    with torch.no_grad():
        for _ in range(10):
            model_cpu(dummy_cpu)

    times_cpu = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            model_cpu(dummy_cpu)
            times_cpu.append(time.perf_counter() - start)

    print(f"\n[Латентность CPU (batch=1)]")
    print(f"  Среднее: {np.mean(times_cpu)*1000:.2f} ms")
    print(f"  FPS:      {1/np.mean(times_cpu):.1f}")

    model.to(device)

    return {
        'total_params': total_params,
        'size_mb': size_mb,
        'gpu_ms': np.mean(times_gpu) * 1000,
        'cpu_ms': np.mean(times_cpu) * 1000,
        'fps_gpu': 1 / np.mean(times_gpu),
        'fps_cpu': 1 / np.mean(times_cpu),
    }


def main():
    # CNN
    cnn_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_cnn.pth')
    cnn_model = GhostNetV2_FER(num_classes=Config.NUM_CLASSES, dropout=0.3).to(Config.DEVICE)
    cnn_ckpt = torch.load(cnn_path, map_location=Config.DEVICE)
    cnn_model.load_state_dict(cnn_ckpt['model_state'])

    cnn_results = benchmark_model(cnn_model, cnn_path, Config.DEVICE, name="CNN (GhostNetV2)")

    # ViT
    vit_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_vit.pth')
    vit_model = ViT_FER(
        img_size=Config.IMAGE_SIZE,
        patch_size=Config.PATCH_SIZE,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        embed_dim=Config.EMBED_DIM,
        depth=Config.DEPTH,
        num_heads=Config.NUM_HEADS,
        dropout=0.1
    ).to(Config.DEVICE)
    vit_ckpt = torch.load(vit_path, map_location=Config.DEVICE)
    vit_model.load_state_dict(vit_ckpt['model_state'])

    vit_results = benchmark_model(vit_model, vit_path, Config.DEVICE, name="Vision Transformer (ViT)")

    # Итоговая таблица
    print(f"\n{'='*70}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ CNN vs ViT")
    print(f"{'='*70}")
    print(f"{'Метрика':<35} {'CNN':>12} {'ViT':>12}")
    print("-" * 60)
    print(f"{'Всего параметров':<35} {cnn_results['total_params']:>12,} {vit_results['total_params']:>12,}")
    print(f"{'Размер файла (MB)':<35} {cnn_results['size_mb']:>12.2f} {vit_results['size_mb']:>12.2f}")
    print(f"{'Латентность GPU (ms)':<35} {cnn_results['gpu_ms']:>12.2f} {vit_results['gpu_ms']:>12.2f}")
    print(f"{'Латентность CPU (ms)':<35} {cnn_results['cpu_ms']:>12.2f} {vit_results['cpu_ms']:>12.2f}")
    print(f"{'FPS GPU':<35} {cnn_results['fps_gpu']:>12.1f} {vit_results['fps_gpu']:>12.1f}")
    print(f"{'FPS CPU':<35} {cnn_results['fps_cpu']:>12.1f} {vit_results['fps_cpu']:>12.1f}")


if __name__ == '__main__':
    main()