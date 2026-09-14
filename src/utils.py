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

