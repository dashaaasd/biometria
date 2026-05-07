import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config, get_train_transform_full
from data import build_dataloaders
from cnn_model import GhostNetV2_FER
from utils import set_seed, get_scheduler, run_epoch


def main():
    set_seed(Config.SEED)

    train_loader, val_loader, _ = build_dataloaders(Config)

    model = GhostNetV2_FER(
        num_classes=Config.NUM_CLASSES,
        dropout=0.3,
        img_size=Config.IMAGE_SIZE,
        in_channels=Config.IN_CHANNELS,
    ).to(Config.DEVICE)
    print(f"CNN (GhostNetV2) | Параметры: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.LEARNING_RATE,
                            weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, warmup_epochs=5, total_epochs=Config.NUM_EPOCHS)
    use_amp   = Config.DEVICE.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_acc  = 0.0
    best_path = os.path.join(Config.MODEL_SAVE_PATH, Config.MODEL_CNN_NAME)

    # С этой эпохи включаем RandomErasing (тяжёлая аугментация)
    AUGMENT_EPOCH = 7

    start = time.time()
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f'\nЭпоха {epoch}/{Config.NUM_EPOCHS}  lr={optimizer.param_groups[0]["lr"]:.2e}')

        if epoch == AUGMENT_EPOCH:
            # train_loader.dataset — это Subset; .dataset — ImageFolder с трансформом
            train_loader.dataset.dataset.transform = get_train_transform_full(Config.IMAGE_SIZE)
            print(f"  RandomErasing включён с эпохи {epoch}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, Config.DEVICE,
                                          use_amp, training=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion,
                                          optimizer, scaler, Config.DEVICE,
                                          use_amp, training=False)
        scheduler.step()

        print(f'  Train: loss={train_loss:.4f}  acc={train_acc:.2f}%')
        print(f'  Val:   loss={val_loss:.4f}  acc={val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc':     best_acc,
                'config': {
                    'img_size':   Config.IMAGE_SIZE,
                    'in_channels': Config.IN_CHANNELS,
                    'dataset':    Config.DATA_PATH,
                }
            }, best_path)
            print(f'  ✓ Сохранена лучшая CNN-модель (val_acc={best_acc:.2f}%)')

    elapsed = time.time() - start
    print(f'\nОбучение CNN завершено за {elapsed / 60:.1f} мин')
    print(f'Лучшая val accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
