import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from config import Config
from data import build_dataloaders
from cnn_model import GhostNetV2_FER
from vit_model import ViT_FER


CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Test'):
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def print_metrics(all_preds, all_labels, model_name):
    test_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n{model_name} | Test accuracy: {test_acc:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    return test_acc


def save_confusion_matrix(all_preds, all_labels, model_name, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='Blues')
    plt.title(f'{model_name} — Confusion Matrix')
    plt.ylabel('Настоящий класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix сохранена: {save_path}")


def main():
    dataset_name = os.path.basename(Config.DATA_PATH)   # кроссплатформенно

    print(f"\n{'='*60}")
    print(f"  ТЕСТИРОВАНИЕ: {Config.DATA_PATH}")
    print(f"{'='*60}")

    _, _, test_loader = build_dataloaders(Config)

    # ── CNN ──────────────────────────────────────────────────────
    cnn_path  = os.path.join(Config.MODEL_SAVE_PATH, Config.MODEL_CNN_NAME)
    cnn_model = GhostNetV2_FER(
        num_classes=Config.NUM_CLASSES,
        dropout=0.3,
        img_size=Config.IMAGE_SIZE,
        in_channels=Config.IN_CHANNELS,
    ).to(Config.DEVICE)
    checkpoint = torch.load(cnn_path, map_location=Config.DEVICE)
    cnn_model.load_state_dict(checkpoint['model_state'])
    print(f"CNN загружена | val_acc={checkpoint['val_acc']:.2f}%")

    cnn_preds, cnn_labels = evaluate(cnn_model, test_loader, Config.DEVICE)
    cnn_acc = print_metrics(cnn_preds, cnn_labels, 'CNN (GhostNetV2)')
    save_confusion_matrix(cnn_preds, cnn_labels, 'CNN-GhostNetV2',
                          f'confusion_matrix_cnn_{dataset_name}.png')

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
    checkpoint = torch.load(vit_path, map_location=Config.DEVICE)
    vit_model.load_state_dict(checkpoint['model_state'])
    print(f"ViT загружена  | val_acc={checkpoint['val_acc']:.2f}%")

    vit_preds, vit_labels = evaluate(vit_model, test_loader, Config.DEVICE)
    vit_acc = print_metrics(vit_preds, vit_labels, 'Vision Transformer (ViT)')
    save_confusion_matrix(vit_preds, vit_labels, 'ViT',
                          f'confusion_matrix_vit_{dataset_name}.png')

    # ── Итоговое сравнение ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ CNN vs ViT")
    print("=" * 60)
    print(f"{'Модель':<25} {'Test Accuracy':>15}")
    print("-" * 40)
    print(f"{'CNN (GhostNetV2)':<25} {cnn_acc:>14.2f}%")
    print(f"{'Vision Transformer':<25} {vit_acc:>14.2f}%")


if __name__ == '__main__':
    main()
