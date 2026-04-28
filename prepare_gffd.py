"""
Разбивка GFFD-2025: объединение Genuine/Fake, разбивка 80/20.
"""
import os
import shutil
import random
from pathlib import Path
from collections import Counter

RAW_DIR   = Path(r'.\data\CroppedDataset')
OUT_DIR   = Path(r'.\data\gffd2025')
TRAIN_RATIO = 0.8
SEED        = 42

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_MAP = {e.lower(): e.lower() for e in EMOTIONS}

def main():
    random.seed(SEED)

    # Создаём структуру
    for split in ['train', 'test']:
        for emo in EMOTION_MAP.values():
            (OUT_DIR / split / emo).mkdir(parents=True, exist_ok=True)

    stats = {'train': Counter(), 'test': Counter()}

    for emo_folder in EMOTIONS:
        emo_lower = emo_folder.lower()
        files = []

        for subtype in ['Genuine', 'Fake']:
            src = RAW_DIR / emo_folder / subtype
            if src.is_dir():
                for fname in os.listdir(src):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        files.append((src / fname, fname))

        random.shuffle(files)
        n_train = int(len(files) * TRAIN_RATIO)

        for (src_path, fname) in files[:n_train]:
            shutil.copy2(src_path, OUT_DIR / 'train' / emo_lower / fname)
        for (src_path, fname) in files[n_train:]:
            shutil.copy2(src_path, OUT_DIR / 'test' / emo_lower / fname)

        stats['train'][emo_lower] = n_train
        stats['test'][emo_lower] = len(files) - n_train

    print("=" * 55)
    print("  GFFD-2025 подготовлен")
    print("=" * 55)
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
    print(f"\nСтруктура: {OUT_DIR.resolve()}")

if __name__ == '__main__':
    main()