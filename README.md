# Сравнение CNN и Vision Transformer для распознавания эмоций на лицах (FER2013)

> Работа оформляется в виде научной статьи (публикация ожидается в 2026 г. при публикации здесь будет ссылка)

## Результаты(переписать)

## Датасеты

### 1. FER2013
- **Классы:** angry, disgust, fear, happy, neutral, sad, surprise
- **Размер:** 48×48 пикселей, градации серого
- **Train:** 22 967 · **Val:** 5 742 · **Test:** 3 589
- **Особенности:** низкое разрешение, шум разметки, дисбаланс классов

### 2. ExpW (Expression in-the-Wild)
- **Классы:** angry, disgust, fear, happy, neutral, sad, surprise
- **Размер:** полноразмерные изображения → 224×224
- **Train:** ~73 435 · **Val:** ~18 358 · **Test:** ~18 358
- **Особенности:** высокое разрешение, in-the-wild условия, сильный дисбаланс

### 3. GFFD-2025 (Genuine and Fake Facial Emotion Dataset)
- **Классы:** angry, disgust, fear, happy, neutral, sad, surprise
  - Каждая эмоция делится на **Genuine** (искренняя) и **Fake** (наигранная) → 14 подклассов
  - В рамках эксперимента Genuine и Fake объединены → 7 классов
- **Размер:** 224×224 пикселей, RGB → градации серого
- **Особенности:** лабораторные условия, контролируемое освещение

## Архитектура(дописать)

### CNN: GhostNetV2
- **Тип:** свёрточная нейронная сеть
- **Особенности:** Ghost-блоки для экономии вычислений, depthwise-свёртки
- **Параметры:** ~1.1 млн
- **Индуктивное смещение:** сильная локальность и трансляционная эквивариантность — хорошо работает на малых данных

### ViT: Compact Vision Transformer
- **Тип:** визуальный трансформер
- **Особенности:** Patch Embedding (8×8 для FER2013, 16×16 для ExpW/GFFD), Multi-Head Self-Attention (8 голов), 6 блоков
- **Параметры:** ~3.2 млн
- **Индуктивное смещение:** отсутствует — все пространственные отношения выучиваются с нуля — лучше масштабируется на больших данных

## Структура проекта

```
FER2013-CNN-vs-ViT/
├── models/
│ ├── init.py # экспорт моделей: GhostNetV2_FER (CNN) и ViT_FER (Vision Transformer)
│ ├── cnn_model.py # GhostNetV2 — легковесная свёрточная архитектура с Ghost-блоками
│ └── vit_model.py # Vision Transformer — патч-эмбеддинг, multi-head attention, трансформер-блоки
├── config.py # пути к данным, гиперпараметры (batch size, lr, эпохи), аугментации
├── data.py # загрузка FER2013, разбиение на train/val/test, создание DataLoader'ов
├── utils.py # вспомогательные функции: set_seed, get_scheduler (warmup + cosine), run_epoch
├── train_cnn.py # скрипт обучения CNN (GhostNetV2) с сохранением лучшей модели
├── train_vit.py # скрипт обучения ViT с сохранением лучшей модели
├── evaluate.py # оценка точности на тестовой выборке + confusion matrix (CNN vs ViT)
├── benchmark.py # сравнение параметров, размера .pth файлов, латентности на GPU и CPU
├── requirements.txt # зависимости: torch, torchvision, numpy, tqdm, scikit-learn, matplotlib, seaborn
└── README.md # документация проекта, таблица результатов, инструкции по воспроизведению
├── prepare_expw.py # подготовка ExpW из label.lst
├── prepare_gffd.py # подготовка GFFD-2025 (объединение Genuine/Fake)
```

## Воспроизведение результатов

**1. Клонировать репозиторий**
```bash
git clone https://github.com/Walfeinick/GhostNetV2-TensorTrain-Project.git
cd biometria
```

**2. Установить зависимости**

Создать виртуальное окружение (рекомендуется Python 3.10):

```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Linux / Mac
```

Сначала установить PyTorch: выбрать команду под свою версию CUDA на [pytorch.org](https://pytorch.org/get-started/locally/).
Пример для CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Затем остальные зависимости:

```bash
pip install -r requirements.txt
```

**3. Подготовить датасет**
Создать папку **data**

```bash
mkdir data
```

Скачать FER2013 с [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), датасет уже разбит на нужную стуктуру:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

Скачать ExpW с [Kaggle](https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset/data) и разложить в следующую структуру:
1)Распаковать архив origin.7z.001–origin.7z.008 в data/expw_raw/:

```bash
cd "data/expression-in-the-wild-expw-dataset/versions/1"
& "C:\Program Files\7-Zip\7z.exe" x origin.7z.001 "-o../../expw_raw"
```

2)Запустить prepare-скрипт:

```bash
python prepare_expw.py
```

Результат: data/expw/train/ и data/expw/test/ с 7 папками.
Итоговая стуктура:

```
data/expw/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

Скачать ExpW с [Mendeley Data](https://data.mendeley.com/datasets/wmfd4p3z32/1) и разложить в следующую структуру:

1)Распаковать архив CroppedAugmentedDataset.zip в data/CroppedDataset/:

2)Запустить prepare-скрипт:

```bash
python prepare_gffd.py
```

Результат: data/gffd2025/train/ и data/gffd2025/test/ с 7 папками.
Итоговая стуктура:

```
data/expw/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

**4. Обучение**

```bash
python train_cnn.py        
python train_vit.py   
```

**5. Оценка**

```bash
python test.py     # точность на тесте + confusion matrix (обе модели)
python benchmark.py    # число параметров, размер модели, латентность GPU/CPU
```

# В config.py: Config = ConfigFER
python train_cnn.py
python train_vit.py
python test.py
python benchmark.py

# В config.py: Config = ConfigExpW
python train_cnn.py
python train_vit.py
python test.py
python benchmark.py

# В config.py: Config = ConfigGFFD
python train_cnn.py
python train_vit.py
python test.py
python benchmark.py

## Требования

- Python 3.10+
- PyTorch 2.6+ с поддержкой CUDA (CPU поддерживается, но обучение будет медленным)
- Полный список зависимостей в `requirements.txt`
