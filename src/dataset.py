## Carregamento e transformações
import os
import torch

from typing import Tuple, List
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Retorna os pipelines de transformação para treino (com data augmentation)
    e validação/teste (apenas normalização).
    """
    # Médias e desvios padrão do ImageNet (padrão para redes de visão)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_test_transform


def load_raw_dataset(data_path: str, transform=None) -> ImageFolder:
    """
    Carrega o dataset a partir de uma pasta organizada por classes (ImageFolder).
    Valida se o caminho existe antes de tentar carregar.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERRO] Diretório não encontrado: {data_path}")

    dataset = ImageFolder(root=data_path, transform=transform)
    
    print(f"[INFO] Dataset carregado de: {data_path}")
    print(f"[INFO] Total de imagens: {len(dataset)}")
    print(f"[INFO] Classes detectadas ({len(dataset.classes)}): {dataset.classes}")
    
    return dataset


def create_dataloader(
    dataset: Dataset, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 2
) -> DataLoader:
    """
    Empacota um Dataset em um DataLoader do PyTorch otimizado para GPU.
    """
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

