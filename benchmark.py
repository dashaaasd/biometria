import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from config import Config
from data import build_dataloaders
from models.cnn_model import GhostNetV2_FER
from models.vit_model import ViT_FER
from models.vit_distill import ViT_Distill 

CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def evaluate_accuracy(model, dataloader, device):
    """Оценка accuracy, F1 и per-class метрик на device."""
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc         = accuracy_score(all_labels, all_preds) * 100
    f1_macro    = f1_score(all_labels, all_preds, average='macro')    * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    cm          = confusion_matrix(all_labels, all_preds)
    per_class   = cm.diagonal() / cm.sum(axis=1) * 100

    return {
        'accuracy':      acc,
        'f1_macro':      f1_macro,
        'f1_weighted':   f1_weighted,
        'per_class_acc': per_class,
        'confusion_matrix': cm,
    }


def measure_gpu_memory(model, input_shape, device):
    if device.type != 'cuda':
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.eval()
    dummy = torch.randn(1, *input_shape).to(device)
    with torch.no_grad():
        model(dummy)
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def benchmark_model(model, model_path, dataloader, device, name='Model'):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Параметры]")
    print(f"  Всего:     {total_params:>10,}")
    print(f"  Обучаемых: {trainable_params:>10,}")

    size_mb = None
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"\n[Размер файла]")
        print(f"  .pth: {size_mb:.2f} MB")

    input_shape = (Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # ── GPU latency ──────────────────────────────────────────────
    model.to(device).eval()
    gpu_mem = measure_gpu_memory(model, input_shape, device)
    print(f"\n[GPU память (batch=1)]  {gpu_mem:.1f} MB")

    dummy = torch.randn(1, *input_shape).to(device)
    with torch.no_grad():                         # warmup
        for _ in range(20):
            model(dummy)

    times_gpu = []
    with torch.no_grad():
        for _ in range(200):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_gpu.append(time.perf_counter() - t0)

    print(f"\n[Латентность GPU (batch=1)]")
    print(f"  Среднее: {np.mean(times_gpu)*1000:.2f} ms  |  FPS: {1/np.mean(times_gpu):.1f}")

    # ── CPU latency ──────────────────────────────────────────────
    # Переносим модель на CPU только для замера, потом возвращаем на device
    model.cpu().eval()
    dummy_cpu = torch.randn(1, *input_shape)
    with torch.no_grad():
        for _ in range(10):
            model(dummy_cpu)

    times_cpu = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            model(dummy_cpu)
            times_cpu.append(time.perf_counter() - t0)

    print(f"\n[Латентность CPU (batch=1)]")
    print(f"  Среднее: {np.mean(times_cpu)*1000:.2f} ms  |  FPS: {1/np.mean(times_cpu):.1f}")

    # ── Accuracy / F1 ─────────────────────────────────────────────
    # evaluate_accuracy сам переносит модель на device
    print(f"\n[Качество на валидации]")
    metrics = evaluate_accuracy(model, dataloader, device)
    print(f"  Accuracy:    {metrics['accuracy']:.2f}%")
    print(f"  F1 Macro:    {metrics['f1_macro']:.2f}%")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.2f}%")
    print(f"\n  Per-class accuracy:")
    for cls, acc in zip(CLASS_NAMES, metrics['per_class_acc']):
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"    {cls:10s}: {bar} {acc:.1f}%")

    return {
        'total_params':     total_params,
        'size_mb':          size_mb,
        'gpu_memory_mb':    gpu_mem,
        'gpu_ms':           np.mean(times_gpu) * 1000,
        'cpu_ms':           np.mean(times_cpu) * 1000,
        'fps_gpu':          1 / np.mean(times_gpu),
        'fps_cpu':          1 / np.mean(times_cpu),
        'accuracy':         metrics['accuracy'],
        'f1_macro':         metrics['f1_macro'],
        'f1_weighted':      metrics['f1_weighted'],
        'per_class_acc':    metrics['per_class_acc'],
        'confusion_matrix': metrics['confusion_matrix'],
    }


def plot_comparison_three(cnn_r, vit_r, vit_dis_r, save_dir, dataset_name):
    """Графики для трёх моделей"""
    if vit_dis_r is None:
        plot_comparison(cnn_r, vit_r, save_dir, dataset_name)
        return
    
    fig = plt.figure(figsize=(16, 5))
    
    # Bar chart
    ax_bar = fig.add_subplot(1, 2, 1)
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    ax_bar.bar(x - w, cnn_r['per_class_acc'], w, label='CNN', alpha=0.8)
    ax_bar.bar(x, vit_r['per_class_acc'], w, label='ViT', alpha=0.8)
    ax_bar.bar(x + w, vit_dis_r['per_class_acc'], w, label='ViT+Distill', alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax_bar.set_ylabel('Accuracy (%)')
    ax_bar.set_title(f'Per-class Accuracy — {dataset_name}')
    ax_bar.legend()
    ax_bar.grid(axis='y', alpha=0.3)
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(CLASS_NAMES), endpoint=False).tolist()
    angles += angles[:1]
    
    ax_rad = fig.add_subplot(1, 2, 2, polar=True)
    for data, label, color in zip(
        [cnn_r['per_class_acc'], vit_r['per_class_acc'], vit_dis_r['per_class_acc']],
        ['CNN', 'ViT', 'ViT+Distill'],
        ['blue', 'orange', 'green']
    ):
        values = data.tolist() + [data[0]]
        ax_rad.plot(angles, values, 'o-', linewidth=1.5, label=label, color=color)
        ax_rad.fill(angles, values, alpha=0.1, color=color)
    
    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(CLASS_NAMES)
    ax_rad.set_title('Per-class Accuracy Radar')
    ax_rad.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    path1 = os.path.join(save_dir, f'per_class_three_{dataset_name}.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {path1}")
    
    # Confusion matrices (3 rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cm, title in zip(
        axes,
        [cnn_r['confusion_matrix'], vit_r['confusion_matrix'], vit_dis_r['confusion_matrix']],
        ['CNN (GhostNetV2)', 'Vision Transformer (ViT)', 'ViT + Distillation']
    ):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    vmin=0, vmax=1, ax=ax)
        ax.set_title(f'{title}\n{dataset_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    path2 = os.path.join(save_dir, f'confusion_matrices_three_{dataset_name}.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {path2}")

def main():
    dataset_name = os.path.basename(Config.DATA_PATH)
    save_dir     = os.path.join('.', 'benchmark_results')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  БЕНЧМАРК: {Config.DATA_PATH}  ({Config.IMAGE_SIZE}×{Config.IMAGE_SIZE})")
    print(f"{'='*70}")

    _, val_loader, _ = build_dataloaders(Config)

    # ── CNN ──────────────────────────────────────────────────────
    cnn_path  = os.path.join(Config.MODEL_SAVE_PATH, Config.MODEL_CNN_NAME)
    cnn_model = GhostNetV2_FER(
        num_classes=Config.NUM_CLASSES,
        dropout=0.3,
        img_size=Config.IMAGE_SIZE,
        in_channels=Config.IN_CHANNELS,
    ).to(Config.DEVICE)
    cnn_model.load_state_dict(
        torch.load(cnn_path, map_location=Config.DEVICE)['model_state']
    )
    cnn_results = benchmark_model(cnn_model, cnn_path, val_loader,
                                  Config.DEVICE, name='CNN (GhostNetV2)')

    # ── ViT ──────────────────────────────────────────────────────
    vit_path  = os.path.join(Config.MODEL_SAVE_PATH, Config.MODEL_VIT_NAME)
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
    vit_model.load_state_dict(
        torch.load(vit_path, map_location=Config.DEVICE)['model_state']
    )
    vit_results = benchmark_model(vit_model, vit_path, val_loader,
                                  Config.DEVICE, name='Vision Transformer (ViT)')
    # ── ViT_Distill ──────────────────────────────────────────────────
    vit_dis_path = os.path.join(Config.MODEL_SAVE_PATH, Config.MODEL_VIT_NAME_DIS)

    if os.path.exists(vit_dis_path):
        vit_dis_model = ViT_Distill(
            img_size=Config.IMAGE_SIZE,
            patch_size=Config.PATCH_SIZE,
            in_channels=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            embed_dim=Config.EMBED_DIM,
            depth=Config.DEPTH,
            num_heads=Config.NUM_HEADS,
            dropout=Config.VIT_DROPOUT
        ).to(Config.DEVICE)
        
        checkpoint = torch.load(vit_dis_path, map_location=Config.DEVICE)
        if 'model_state_dict' in checkpoint:
            vit_dis_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            vit_dis_model.load_state_dict(checkpoint)
        
        vit_dis_results = benchmark_model(vit_dis_model, vit_dis_path, val_loader,
                                        Config.DEVICE, name='ViT + Distillation')
    else:
        print(f"\n⚠️ Модель дистилляции не найдена: {vit_dis_path}")
        vit_dis_results = None

    # ── Итоговая таблица (3 модели) ─────────────────────────────────
    print(f"\n{'='*90}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ: CNN vs ViT vanilla vs ViT + Distillation")
    print(f"{'='*90}")

    # Заголовки
    print(f"{'Метрика':<30} {'CNN':>12} {'ViT':>12} {'ViT+Distill':>14} {'Δ Distill-ViT':>14}")
    print("-" * 90)

    def row3(label, c, v, d, fmt='.2f'):
        diff = d - v if d is not None else 0
        if d is not None:
            if fmt == '.1f':
                print(f"{label:<30} {c:>12.1f} {v:>12.1f} {d:>14.1f} {diff:>+14.1f}")
            else:
                print(f"{label:<30} {c:>12.2f} {v:>12.2f} {d:>14.2f} {diff:>+14.2f}")
        else:
            print(f"{label:<30} {c:>12.2f} {v:>12.2f} {'N/A':>14} {'N/A':>14}")
    def row3_int(label, c, v, d):
        if d is not None:
            print(f"{label:<30} {c:>12,} {v:>12,} {d:>14,}")
        else:
            print(f"{label:<30} {c:>12,} {v:>12,} {'N/A':>14}")

    print("\n  --- Эффективность ---")
    row3_int("Параметры", cnn_results['total_params'], vit_results['total_params'], 
            vit_dis_results['total_params'] if vit_dis_results else None)

    if cnn_results['size_mb'] and vit_dis_results and vit_dis_results['size_mb']:
        row3("Размер файла (MB)", cnn_results['size_mb'], vit_results['size_mb'], 
            vit_dis_results['size_mb'])

    row3("GPU память (MB)", cnn_results['gpu_memory_mb'], vit_results['gpu_memory_mb'],
        vit_dis_results['gpu_memory_mb'] if vit_dis_results else None, fmt='.1f')
    row3("Латентность GPU (ms)", cnn_results['gpu_ms'], vit_results['gpu_ms'],
        vit_dis_results['gpu_ms'] if vit_dis_results else None)
    row3("Латентность CPU (ms)", cnn_results['cpu_ms'], vit_results['cpu_ms'],
        vit_dis_results['cpu_ms'] if vit_dis_results else None)
    row3("FPS GPU", cnn_results['fps_gpu'], vit_results['fps_gpu'],
        vit_dis_results['fps_gpu'] if vit_dis_results else None)
    row3("FPS CPU", cnn_results['fps_cpu'], vit_results['fps_cpu'],
        vit_dis_results['fps_cpu'] if vit_dis_results else None)

    print("\n  --- Качество ---")
    row3("Accuracy (%)", cnn_results['accuracy'], vit_results['accuracy'],
        vit_dis_results['accuracy'] if vit_dis_results else None)
    row3("F1 Macro (%)", cnn_results['f1_macro'], vit_results['f1_macro'],
        vit_dis_results['f1_macro'] if vit_dis_results else None)
    row3("F1 Weighted (%)", cnn_results['f1_weighted'], vit_results['f1_weighted'],
        vit_dis_results['f1_weighted'] if vit_dis_results else None)

    # Per-class сравнение
    if vit_dis_results:
        print("\n  --- Per-class Accuracy ---")
        print(f"  {'Class':<12} {'CNN':>8} {'ViT':>8} {'ViT+Distill':>12} {'Best':>8}")
        print("-" * 55)
        for i, cls in enumerate(CLASS_NAMES):
            c = cnn_results['per_class_acc'][i]
            v = vit_results['per_class_acc'][i]
            d = vit_dis_results['per_class_acc'][i]
            best = max(c, v, d)
            best_name = 'CNN' if best == c else ('ViT' if best == v else 'Distill')
            print(f"  {cls:<12} {c:>7.1f}% {v:>7.1f}% {d:>11.1f}%  → {best_name}")
    plot_comparison_three(cnn_results, vit_results, vit_dis_results, save_dir, dataset_name)

if __name__ == '__main__':
    main()
