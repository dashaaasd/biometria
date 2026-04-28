"""
Создание degraded-версий датасетов (48×48).
Запуск: python prepare_degraded.py
"""
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

TASKS = [
    {'src': r'.\data\expw',      'dst': r'.\data\expw_48',    'name': 'ExpW_48'},
    {'src': r'.\data\gffd2025',  'dst': r'.\data\gffd2025_48','name': 'GFFD-2025_48'},
]

TARGET_SIZE = 48

def main():
    for task in TASKS:
        src_dir = Path(task['src'])
        dst_dir = Path(task['dst'])
        
        if not src_dir.is_dir():
            print(f"Пропущен: {src_dir} не найден")
            continue
        
        print(f"\n{'='*50}")
        print(f"  {task['name']}: 224×224 → {TARGET_SIZE}×{TARGET_SIZE}")
        print(f"{'='*50}")
        
        for split in ['train', 'test']:
            src_split = src_dir / split
            for emotion in os.listdir(src_split):
                src_emo = src_split / emotion
                if not src_emo.is_dir():
                    continue
                
                dst_emo = dst_dir / split / emotion
                dst_emo.mkdir(parents=True, exist_ok=True)
                
                files = [f for f in os.listdir(src_emo)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for fname in tqdm(files, desc=f'{split}/{emotion}', leave=False):
                    img = Image.open(src_emo / fname).convert('L')
                    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
                    img.save(dst_emo / fname)
        
        print(f"Готово: {dst_dir}")
    
    print("\nВсе degraded-версии созданы.")

if __name__ == '__main__':
    main()