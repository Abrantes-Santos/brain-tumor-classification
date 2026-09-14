##Loop de treino, otimizador e loss

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# importações internas dos seus módulos
from src.model import BrainTumorCNN
from src.dataset import get_dataloaders
