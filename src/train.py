##Loop de treino, otimizador e loss


import argparse
import os
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.model import BrainTumorCNN
from src.utils import get_device, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Executa uma época completa de treinamento."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Executa a avaliação do modelo no conjunto de validação."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def train(args: argparse.Namespace) -> Dict[str, List[float]]:
    """Pipeline completo de treinamento com salvamento de checkpoint."""
    set_seed(args.seed)
    device = get_device()

    # 1. Pipeline de dados
    train_loader, val_loader, class_names = build_dataloaders(
        data_path=args.data_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # 2. Inicialização do Modelo
    model = BrainTumorCNN(num_classes=len(class_names), dropout_rate=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Histórico de métricas
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "best_model.pth")

    print(f"\n[INFO] Iniciando treinamento por {args.epochs} épocas...")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Checkpoint: salva o estado com melhor acurácia de validação
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names
            }, checkpoint_path)
            saved_tag = "[NOVO MELHOR]"
        else:
            saved_tag = ""

        print(
            f"Época [{epoch+1:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% {saved_tag}"
        )

    print(f"\n[SUCESSO] Treinamento concluído. Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"[INFO] Modelo salvo em: {checkpoint_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento CNN Brain Tumor")
    parser.add_argument("--data_path", type=str, default="/content/drive/MyDrive/Training", help="Caminho do dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Diretório para salvar os pesos")
    parser.add_argument("--epochs", type=int, default=40, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--img_size", type=int, default=128, help="Resolução das imagens")
    parser.add_argument("--lr", type=float, default=0.001, help="Taxa de aprendizado")
    parser.add_argument("--dropout", type=float, default=0.4, help="Taxa de dropout")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proporção de treino")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers do DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória")

    args = parser.parse_args()
    train(args)
