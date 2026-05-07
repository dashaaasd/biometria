import os
import torch
from torchvision import transforms

# КОНФИГУРАЦИИ ДЛЯ ТРЁХ ДАТАСЕТОВ

class ConfigFER:
    """FER2013 (48×48)"""
    DATA_PATH = r'.\data\fer2013'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS = 1
    IMAGE_SIZE = 48
    NUM_CLASSES = 7
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 3e-4
    PATCH_SIZE = 8
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0 if os.name == 'nt' else 4
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME = 'best_model_cnn_fer.pth'
    MODEL_VIT_NAME = 'best_model_vit_fer.pth'


class ConfigExpW:
    """ExpW (224×224)"""
    DATA_PATH = r'.\data\expw'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS = 1
    IMAGE_SIZE = 224
    NUM_CLASSES = 7
    BATCH_SIZE = 32
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 3e-4
    PATCH_SIZE = 16
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0 if os.name == 'nt' else 4
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME = 'best_model_cnn_expw.pth'
    MODEL_VIT_NAME = 'best_model_vit_expw.pth'


class ConfigGFFD:
    """GFFD-2025 (224×224)"""
    DATA_PATH = r'.\data\gffd2025'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS = 1
    IMAGE_SIZE = 224
    NUM_CLASSES = 7
    BATCH_SIZE = 32
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 3e-4
    PATCH_SIZE = 16
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0 if os.name == 'nt' else 4
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME = 'best_model_cnn_gffd.pth'
    MODEL_VIT_NAME = 'best_model_vit_gffd.pth'

class ConfigExpW_48:
    """ExpW degraded (48×48)"""
    DATA_PATH = r'.\data\expw_48'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS = 1
    IMAGE_SIZE = 48
    NUM_CLASSES = 7
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 3e-4
    PATCH_SIZE = 8
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0 if os.name == 'nt' else 4
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME = 'best_model_cnn_expw_48.pth'
    MODEL_VIT_NAME = 'best_model_vit_expw_48.pth'


class ConfigGFFD_48:
    """GFFD-2025 degraded (48×48)"""
    DATA_PATH = r'.\data\gffd2025_48'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS = 1
    IMAGE_SIZE = 48
    NUM_CLASSES = 7
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 3e-4
    PATCH_SIZE = 8
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0 if os.name == 'nt' else 4
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME = 'best_model_cnn_gffd_48.pth'
    MODEL_VIT_NAME = 'best_model_vit_gffd_48.pth'

# ФУНКЦИИ ТРАНСФОРМОВ

def get_train_transform(image_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_train_transform_full(image_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])


def get_val_transform(image_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

# АКТИВНЫЙ КОНФИГ (меняй здесь)
Config = ConfigFER        # FER2013
#Config = ConfigExpW     # ExpW
#Config = ConfigGFFD     # GFFD-2025
# Config = ConfigExpW_48 #ухудшенный ExpW
# Config = ConfigGFFD_48 #ухудшенный GFFD-2025

os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)