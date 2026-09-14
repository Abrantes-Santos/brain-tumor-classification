##Métricas e visualização
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Fixa a semente para garantir reprodutibilidade em Python, NumPy e PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # caso use múltiplas GPUs
    
    # Garante comportamento determinístico nos algoritmos convolucionais da cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Semente global configurada para: {seed}")

def get_device() -> torch.device:
    """
    Retorna o dispositivo disponível (CUDA ou CPU) e exibe informações da GPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Dispositivo em uso: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("[INFO] Dispositivo em uso: CPU")
    return device


from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Reverte a normalização ImageNet aplicada nos tensores para permitir visualização RGB.
    Espera um tensor com formato (C, H, W).
    """
    img = tensor.cpu().detach().numpy().transpose((1, 2, 0))
    img = np.array(std) * img + np.array(mean)
    return np.clip(img, 0, 1)


def plot_batch_samples(
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Gera um grid com amostras do DataLoader, exibindo a imagem desnormalizada
    e seu respectivo rótulo de classe. Opcionalmente salva a figura em disco.
    """
    images, labels = next(iter(dataloader))
    num_samples = min(num_samples, len(images))
    
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    fig.suptitle("Brain MRI - Amostras do Dataset", fontsize=14, fontweight="bold")
    
    # Garante que axes seja iterável como array 2D
    if rows == 1:
        axes = np.array([axes])
        
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        
        if i < num_samples:
            ax.imshow(denormalize(images[i]))
            label_name = class_names[labels[i].item()]
            ax.set_title(f"Classe: {label_name}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Gráfico salvo em: {save_path}")
        
    plt.show()

