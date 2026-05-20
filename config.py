import os
import torch
from torchvision import transforms

# КОНФИГУРАЦИИ ДЛЯ ДАТАСЕТОВ
#
# Каналы:
#   IN_CHANNELS=1 — FER2013 (серый), ExpW_48 и GFFD_48 (деградированы в grayscale)
#   IN_CHANNELS=3 — ExpW и GFFD (цветные RGB)

class ConfigFER:
    """FER2013 (48x48, grayscale)"""
    DATA_PATH       = r'.\data\fer2013'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS     = 1
    IMAGE_SIZE      = 48
    NUM_CLASSES     = 7
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 50
    LEARNING_RATE   = 1e-4
    WEIGHT_DECAY    = 3e-4
    PATCH_SIZE      = 8
    EMBED_DIM       = 256
    DEPTH           = 4
    NUM_HEADS       = 8
    VAL_SPLIT       = 0.2
    NUM_WORKERS     = 4
    SEED            = 42
    DROPOUT      = 0.3    
    VIT_DROPOUT  = 0.1
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME  = 'best_model_cnn_fer.pth'
    MODEL_VIT_NAME  = 'best_model_vit_fer.pth'
    CLASS_WEIGHTS   = 'auto'


class ConfigExpW:
    """ExpW (224x224, RGB)"""
    DATA_PATH       = r'.\data\ExpWConv'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS     = 3
    IMAGE_SIZE      = 224
    NUM_CLASSES     = 7
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 50
    LEARNING_RATE   = 1e-4
    WEIGHT_DECAY    = 3e-4
    PATCH_SIZE      = 16
    EMBED_DIM       = 256
    DEPTH           = 4
    NUM_HEADS       = 8
    VAL_SPLIT       = 0.2
    NUM_WORKERS     = 4
    DROPOUT      = 0.5    
    VIT_DROPOUT  = 0.1    
    SEED            = 42
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME  = 'best_model_cnn_expw.pth'
    MODEL_VIT_NAME  = 'best_model_vit_expw.pth'
    CLASS_WEIGHTS   = 'auto'

'''
class ConfigGFFD:
    """GFFD-2025 (224x224, RGB)"""
    DATA_PATH       = r'.\data\gffd2025'
    MODEL_SAVE_PATH = r'.\models'
    IN_CHANNELS     = 3
    IMAGE_SIZE      = 224
    NUM_CLASSES     = 7
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 50
    LEARNING_RATE   = 1e-4
    WEIGHT_DECAY    = 3e-4
    PATCH_SIZE      = 16
    EMBED_DIM       = 256
    DEPTH           = 4
    NUM_HEADS       = 8
    VAL_SPLIT       = 0.2
    NUM_WORKERS     = 4
    SEED            = 42
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_CNN_NAME  = 'best_model_cnn_gffd.pth'
    MODEL_VIT_NAME  = 'best_model_vit_gffd.pth'
    CLASS_WEIGHTS   = 'auto'
'''

# Трансформы — отдельные для grayscale (in_channels=1) и RGB (in_channels=3)
# RGB нормализуется по статистике ImageNet

def get_train_transform(image_size, in_channels=1):
    if in_channels == 1:
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
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_train_transform_full(image_size, in_channels=1):
    if in_channels == 1:
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
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ])


def get_val_transform(image_size, in_channels=1):
    if in_channels == 1:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# Активный конфиг (меняй здесь)
Config = ConfigFER
#Config = ConfigExpW

os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)