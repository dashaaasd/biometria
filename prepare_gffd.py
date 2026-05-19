"""
Разбивка GFFD-2025: объединение CroppedDataset + raw (Fake/Genuine из обеих папок).
Каждая папка-источник имеет префикс, чтобы избежать перезаписи.
"""
import os
import shutil
import random
from pathlib import Path
from collections import Counter

# Два источника с одинаковой структурой (Эмоция/Fake, Эмоция/Genuine)
SOURCE_DIRS = [
    Path(r'.\data\CroppedDataset'),
    Path(r'.\data\Raw'),
]

OUT_DIR   = Path(r'.\data\gffd2025')
TRAIN_RATIO = 0.8
SEED        = 42

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_MAP = {e.lower(): e.lower() for e in EMOTIONS}

def collect_files(source_dir):
    """
    Собирает все файлы из папки-источника.
    Возвращает: { emotion: [(src_path, subtype, fname), ...] }
    """
    data = {e.lower(): [] for e in EMOTIONS}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    for emo in EMOTIONS:
        emo_lower = emo.lower()
        for subtype in ['Genuine', 'Fake']:
            src = source_dir / emo / subtype
            if src.is_dir():
                for fname in os.listdir(src):
                    if Path(fname).suffix.lower() in exts:
                        data[emo_lower].append((src / fname, subtype, fname))
    return data

def main():
    random.seed(SEED)

    # Очищаем выходную папку
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ['train', 'test']:
        for emo in EMOTION_MAP.values():
            (OUT_DIR / split / emo).mkdir(parents=True, exist_ok=True)

    stats_source = {src.name: Counter() for src in SOURCE_DIRS}
    stats_train  = Counter()
    stats_test   = Counter()

    for emo in EMOTIONS:
        emo_lower = emo.lower()
        all_files = []
        seen_names = set()  # для отслеживания дубликатов имён из разных источников

        for src_dir in SOURCE_DIRS:
            data = collect_files(src_dir)
            src_alias = src_dir.name  # 'CroppedDataset' или 'raw'
            
            for src_path, subtype, fname in data[emo_lower]:
                # Уникальное имя: префикс из названия источника + подтип
                prefix = f"{src_alias}_{subtype.lower()}"
                new_name = f"{prefix}_{fname}"
                
                # На случай, если всё же совпало (маловероятно, но перестрахуемся)
                counter = 1
                base, ext = os.path.splitext(new_name)
                while new_name in seen_names:
                    new_name = f"{base}_{counter}{ext}"
                    counter += 1
                seen_names.add(new_name)
                
                all_files.append((src_path, new_name))
                stats_source[src_alias][emo_lower] += 1

        if not all_files:
            print(f"⚠️  {emo_folder}: нет файлов ни в одном источнике!")
            continue

        random.shuffle(all_files)
        n_train = int(len(all_files) * TRAIN_RATIO)

        for src_path, fname in all_files[:n_train]:
            dst = OUT_DIR / 'train' / emo_lower / fname
            shutil.copy2(src_path, dst)
            stats_train[emo_lower] += 1

        for src_path, fname in all_files[n_train:]:
            dst = OUT_DIR / 'test' / emo_lower / fname
            shutil.copy2(src_path, dst)
            stats_test[emo_lower] += 1

        # Проверка
        actual = len(list((OUT_DIR / 'train' / emo_lower).glob('*'))) + \
                 len(list((OUT_DIR / 'test' / emo_lower).glob('*')))
        expected = len(all_files)
        if actual != expected:
            print(f"⚠️  {emo}: ожидалось {expected}, скопировано {actual}!")

    # ============================================
    # ОТЧЁТ
    # ============================================
    print("=" * 70)
    print("  GFFD-2025 — объединение CroppedDataset + raw")
    print("=" * 70)

    print(f"\n📂 Источники:")
    for src_dir in SOURCE_DIRS:
        src_alias = src_dir.name
        total_src = sum(stats_source[src_alias].values())
        if src_dir.exists():
            print(f"   ✅ {src_alias}: {total_src} файлов")
        else:
            print(f"   ❌ {src_alias}: не найден!")
    
    print(f"\n📊 Разбивка по классам:")
    print(f"{'Эмоция':<12} {'Train':>8} {'Test':>8} {'Всего':>8}")
    print("-" * 40)
    total_train, total_test = 0, 0
    for emo in EMOTION_MAP.values():
        tr = stats_train[emo]
        te = stats_test[emo]
        total_train += tr
        total_test  += te
        print(f"{emo:<12} {tr:>8} {te:>8} {tr+te:>8}")
    print("-" * 40)
    print(f"{'ИТОГО':<12} {total_train:>8} {total_test:>8} {total_train+total_test:>8}")

    # Детализация по источникам внутри каждой эмоции
    print(f"\n🔍 Детализация по источникам:")
    print(f"{'Эмоция':<12} {'Источник':<20} {'Файлов':>8}")
    print("-" * 42)
    for emo in EMOTION_MAP.values():
        for src_dir in SOURCE_DIRS:
            src_alias = src_dir.name
            cnt = stats_source[src_alias].get(emo, 0)
            if cnt > 0:
                print(f"{emo:<12} {src_alias:<20} {cnt:>8}")
    print(f"\n📁 Выходная папка: {OUT_DIR.resolve()}")

if __name__ == '__main__':
    main()