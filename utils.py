import math
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_scheduler(optimizer, warmup_epochs=6, total_epochs=50):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, training: bool, num_classes: int = 7):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    # Для подсчета F1
    all_preds = []
    all_labels = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc='Train' if training else 'Val  ', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Сохраняем предсказания и метки для F1
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Считаем текущий F1 для отображения
            if len(all_preds) > 0:
                current_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                pbar.set_postfix(loss=f'{loss.item():.4f}', 
                               f1=f'{current_f1:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}', 
                               acc=f'{100*correct/total:.1f}%')

    # Финальный подсчет F1 за всю эпоху
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return total_loss / len(loader), 100 * correct / total, epoch_f1 * 100  # F1 в процентах


def calc_f1(true_labels, preds, num_classes: int = 7):
    """Отдельная функция для расчета F1"""
    return f1_score(true_labels, preds, average='weighted', zero_division=0)