"""
Разбивка ExpW по папкам ImageFolder.
Читает label.lst и копирует изображения в data/expw/train/ и data/expw/test/.
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter


# ─── НАСТРОЙКИ ─────────────────────────────────────────────
RAW_DIR   = Path(r'.\data\expression-in-the-wild-expw-dataset\expw_raw\origin')          # сюда распаковали все jpg
LABEL_LST = Path(r'.\data\expression-in-the-wild-expw-dataset\label.lst')
OUT_DIR   = Path(r'.\data\expw')              # итоговая структура
TRAIN_RATIO = 0.8
SEED        = 42

# Метки из label.lst → название папки
EMOTION_MAP = {
    '0': 'angry',
    '1': 'disgust',
    '2': 'fear',
    '3': 'happy',
    '4': 'sad',
    '5': 'surprise',
    '6': 'neutral',
}
# ────────────────────────────────────────────────────────────


def main():
    random.seed(SEED)

    if not LABEL_LST.exists():
        raise FileNotFoundError(f"Нет label.lst: {LABEL_LST}")
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Нет папки с изображениями: {RAW_DIR}")

    # Создаём структуру папок
    for split in ['train', 'test']:
        for emo in EMOTION_MAP.values():
            (OUT_DIR / split / emo).mkdir(parents=True, exist_ok=True)

    # Читаем label.lst и группируем файлы по эмоциям
    emotion_files = {emo: [] for emo in EMOTION_MAP.values()}
    missing = 0

    with open(LABEL_LST, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            filename = parts[0]
            label    = parts[1]  # 0–6

            emotion = EMOTION_MAP.get(label)
            if emotion is None:
                continue

            src_path = RAW_DIR / filename
            if src_path.exists():
                emotion_files[emotion].append(filename)
            else:
                missing += 1

    # Разбиваем и копируем
    stats = {'train': Counter(), 'test': Counter()}

    for emo, files in emotion_files.items():
        random.shuffle(files)
        n_train = int(len(files) * TRAIN_RATIO)

        for fname in files[:n_train]:
            shutil.copy2(RAW_DIR / fname, OUT_DIR / 'train' / emo / fname)
        for fname in files[n_train:]:
            shutil.copy2(RAW_DIR / fname, OUT_DIR / 'test' / emo / fname)

        stats['train'][emo] = n_train
        stats['test'][emo]  = len(files) - n_train

    # Итоговая таблица
    print("=" * 65)
    print("  ExpW подготовлен")
    print("=" * 65)
    print(f"Пропущено (файл не найден): {missing}")
    print()
    print(f"{'Эмоция':<15} {'Train':>8} {'Test':>8} {'Всего':>8}")
    print("-" * 45)
    total_train, total_test = 0, 0
    for emo in EMOTION_MAP.values():
        tr = stats['train'][emo]
        te = stats['test'][emo]
        total_train += tr
        total_test  += te
        print(f"{emo:<15} {tr:>8} {te:>8} {tr+te:>8}")
    print("-" * 45)
    print(f"{'ИТОГО':<15} {total_train:>8} {total_test:>8} {total_train+total_test:>8}")
    print(f"\nСтруктура сохранена в: {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()