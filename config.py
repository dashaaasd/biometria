import os
import torch
from torchvision import transforms


class Config:
    DATA_PATH = r'.\data\fer2013'
    MODEL_SAVE_PATH = r'.\models'

    IN_CHANNELS = 1
    IMAGE_SIZE = 48
    NUM_CLASSES = 7

    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 3e-4

    # ViT параметры
    PATCH_SIZE = 8
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8

    VAL_SPLIT = 0.2
    NUM_WORKERS = 0 if os.name == 'nt' else 4
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

# Трансформы
train_transform_base = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_transform_full = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])