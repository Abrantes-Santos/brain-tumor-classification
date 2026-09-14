# Brain MRI Tumor Classification with Deep Learning (PyTorch)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pipeline modular de Deep Learning construído em **PyTorch** para detecção e classificação multi-classe de tumores cerebrais a partir de exames de Ressonância Magnética (MRI). O modelo categoriza os exames em 4 classes clínicas com suporte completo a treinamento e inferência via linha de comando (CLI).

---

## 🎯 Classes Diagnosticadas

* **Glioma (`glioma_tumor`)**
* **Meningioma (`meningioma_tumor`)**
* **Adenoma Pituitário (`pituitary_tumor`)**
* **Sem Tumor / Controle Saudável (`no_tumor`)**

---

## 🏗️ Arquitetura da Rede (`BrainTumorCNN`)

A arquitetura foi projetada para processar tensores normalizados de dimensão $(3, 128, 128)$, utilizando camadas de convolução 2D, Batch Normalization para estabilização de gradientes e regularização por Dropout:

```text
Entrada: Tensor (3, 128, 128)
│
├── [Bloco Conv 1]: Conv2d(3 -> 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)  => (32, 64, 64)
├── [Bloco Conv 2]: Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) => (64, 32, 32)
├── [Bloco Conv 3]: Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) => (128, 16, 16)
│
├── [Classificador]: Flatten() => Vetor denso de 32.768 dimensões
├── Linear(32.768 -> 256) -> ReLU -> Dropout(p=0.4)
└── Linear(256 -> 4) => Logits de Saída


📁 Estrutura do Projeto

brain-tumor-mri-pytorch/
├── checkpoints/            # Modelos salvos (.pt / .pth - ignorados no git)
├── data/                   # Diretório de imagens organizado por classes (ignorado no git)
├── reports/                # Métricas e curvas geradas
│   └── figures/
├── src/                    # Código-fonte modular
│   ├── __init__.py
│   ├── dataset.py          # Transforms, wrappers e DataLoaders
│   ├── model.py            # Definição da classe BrainTumorCNN
│   ├── predict.py          # Script de inferência em imagens avulsas (CLI)
│   ├── train.py            # Pipeline de treino, validação e checkpointing
│   └── utils.py            # Sementes (seeds), desnormalização e plots
├── .gitignore
├── requirements.txt        # Dependências do projeto
└── README.md

💻 Guia de Execução Passo a Passo
O repositório foi projetado de forma agnóstica de ambiente. Escolha uma das opções abaixo para rodar os scripts:

Opção 1: Executar na Nuvem via Google Colab (Recomendado / Sem Instalação Local)
Ideal para quem não possui GPU dedicada no computador:

Abra um novo notebook no Google Colab.

Ative a GPU gratuita: Ambiente de execução > Alterar tipo de ambiente de execução > T4 GPU > Salvar.

Em uma célula de código, clone o repositório e instale as dependências:

Bash
!git clone [https://github.com/SEU-USUARIO/brain-tumor-mri-pytorch.git](https://github.com/SEU-USUARIO/brain-tumor-mri-pytorch.git)
%cd brain-tumor-mri-pytorch
!pip install -r requirements.txt
Monte o Google Drive para apontar suas imagens:

Python
from google.colab import drive
drive.mount('/content/drive')
Inicie o treinamento via terminal do Colab:

Bash
!python -m src.train --data_path "/content/drive/MyDrive/Training" --epochs 40 --batch_size 32
