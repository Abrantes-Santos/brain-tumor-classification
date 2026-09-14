#Definição da arquitetura da rede
import torch
import torch.nn as nn


class BrainTumorCNN(nn.Module):
    """
    CNN para classificação de ressonâncias magnéticas cerebrais.
    Entrada esperada: Tensor com formato (Batch, 3, 128, 128).
    """
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.4):
        super(BrainTumorCNN, self).__init__()

        self.features = nn.Sequential(
            # Bloco 1: (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloco 2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloco 3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """
    Calcula e retorna o número total de parâmetros treináveis do modelo.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Teste de sanidade do tensor de entrada e saída
    dummy_input = torch.randn(2, 3, 128, 128)
    model = BrainTumorCNN(num_classes=4)
    output = model(dummy_input)
    
    print(f"Formato da saída: {output.shape}")  # Deve exibir: torch.Size([2, 4])
    print(f"Total de parâmetros treináveis: {count_parameters(model):,}")
