# Сравнение CNN и Vision Transformer для распознавания эмоций на лицах (FER2013)

> Работа оформляется в виде научной статьи (публикация ожидается в 2026 г. при публикации здесь будет ссылка)

## Результаты(переписать)

| Метрика | GhostNetV2-Base | TT-GhostNetV2 | Разница |
|---------|----------------|---------------|---------|
| Test Accuracy | 61.87% | **63.32%** | +1.45% |
| Параметры (всего) | 491,751 | **379,495** | −22.8% |
| Параметры FC-слоя | 123,008 | **10,752** | −11.4x |
| Размер модели | 6.14 MB | **4.79 MB** | −22% |
| Латентность GPU | 3.14 ms | 3.30 ms | ~equal |
| Латентность CPU | 10.08 ms | 10.42 ms | ~equal |

TT-разложение сжало FC-слой в **11.4 раза** при этом точность на тесте
выросла на 1.45% - за счёт регуляризирующего эффекта разложения.

## Датасет

**FER2013** 7 классов эмоций: angry, disgust, fear, happy, neutral, sad, surprise.  
Train: 22,968 · Val: 5,741 · Test: 7,178

## Архитектура(дописать)
Для эксперимента оригинальный FC-слой `Linear(960 → 128)` заменён на `TTLinear`, это
три малых тензорных ядра с рангом `r=16`:

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
```

## Воспроизведение результатов

**1. Клонировать репозиторий**
```bash
git clone https://github.com/Walfeinick/GhostNetV2-TensorTrain-Project.git
cd GhostNetV2-TensorTrain-Project
```

**2. Установить зависимости**

Сначала установить PyTorch: выбрать команду под свою версию CUDA на [pytorch.org](https://pytorch.org/get-started/locally/).
Пример для CUDA 12.4:
```bash

```
Затем остальные зависимости:
```bash
pip install -r requirements.txt
```

**3. Подготовить датасет**

Скачать FER2013 с [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) и разложить в следующую структуру:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
Скачать ExpW с [Kaggle](https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset/data) и разложить в следующую структуру:
```
data/expression-in-the-wild-expw-dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
```
После этого обновить `DATA_PATH` и `MODEL_SAVE_PATH` в `config.py` под свои локальные пути.

**4. Обучение**
```bash
python train_cnn.py        
python train_vit.py   
```

**5. Оценка**
```bash
python evaluate.py     # точность на тесте + confusion matrix (обе модели)
python benchmark.py    # число параметров, размер модели, латентность GPU/CPU
```

## Требования

- Python 3.10+
- PyTorch 2.6+ с поддержкой CUDA (CPU поддерживается, но обучение будет медленным)
- Полный список зависимостей в `requirements.txt`
