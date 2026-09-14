import os
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ApplyTransform(Dataset):
    """
    Wrapper para aplicar transformações específicas em subconjuntos 
    gerados pelo torch.utils.data.random_split.
    """
    def __init__(self, subset: Dataset, transform: transforms.Compose = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


def get_transforms(img_size: int = 128) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Retorna os pipelines de transformação para treino (com data augmentation)
    e validação/teste (apenas redimensionamento e normalização).
    Entrada gerada: tensores de dimensões (3, img_size, img_size).
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


def build_dataloaders(
    data_path: str,
    img_size: int = 128,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Carrega o dataset, realiza o split estratificado via seed e retorna 
    os DataLoaders de treino e validação prontos para o modelo.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERRO] Caminho não encontrado: {data_path}")

    # Carrega sem transform inicial para que ApplyTransform cuide das regras
    raw_dataset = ImageFolder(root=data_path)
    classes = raw_dataset.classes

    total_len = len(raw_dataset)
    train_size = int(train_split * total_len)
    val_size = total_len - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        raw_dataset, [train_size, val_size], generator=generator
    )

    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_dataset = ApplyTransform(train_subset, transform=train_tf)
    val_dataset = ApplyTransform(val_subset, transform=eval_tf)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"[INFO] Classes ({len(classes)}): {classes}")
    print(f"[INFO] Treino: {len(train_dataset)} amostras | Validação: {len(val_dataset)} amostras")

    return train_loader, val_loader, classes
