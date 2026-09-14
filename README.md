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

A arquitetura foi projetada para processar tensores normalizados de dimensão (3, 128, 128), utilizando camadas de convolução 2D, Batch Normalization para estabilização de gradientes e regularização por Dropout:

Entrada: Tensor (3, 128, 128)
│
├── [Bloco Conv 1]: Conv2d(3 -> 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)  => (32, 64, 64)
├── [Bloco Conv 2]: Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) => (64, 32, 32)
├── [Bloco Conv 3]: Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) => (128, 16, 16)
│
├── [Classificador]: Flatten() => Vetor denso de 32.768 dimensões
├── Linear(32.768 -> 256) -> ReLU -> Dropout(p=0.4)
└── Linear(256 -> 4) => Logits de Saída

* **Total de parâmetros treináveis:** ~8,4 milhões
* **Otimizador:** Adam (lr = 0.001)
* **Função de Perda:** Cross-Entropy Loss

---

## 📁 Estrutura do Projeto

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

---

## 💻 Guia de Execução Passo a Passo

O repositório foi projetado de forma agnóstica de ambiente. Escolha uma das opções abaixo para rodar os scripts:

### Opção 1: Executar na Nuvem via Google Colab (Recomendado / Sem Instalação Local)

Ideal para quem não possui GPU dedicada no computador:

1. Abra um novo notebook no Google Colab (https://colab.research.google.com).
2. Ative a GPU gratuita: **Ambiente de execução > Alterar tipo de ambiente de execução > T4 GPU > Salvar**.
3. Em uma célula de código, clone o repositório e instale as dependências:
   !git clone https://github.com/SEU-USUARIO/brain-tumor-mri-pytorch.git
   %cd brain-tumor-mri-pytorch
   !pip install -r requirements.txt
4. Monte o Google Drive para apontar suas imagens:
   from google.colab import drive
   drive.mount('/content/drive')
5. Inicie o treinamento via terminal do Colab:
   !python -m src.train --data_path "/content/drive/MyDrive/Training" --epochs 40 --batch_size 32

---

### Opção 2: Executar Localmente (Linux, macOS ou Windows)

#### 1. Pré-requisitos
* Git instalado
* Python 3.10+ instalado

#### 2. Clonar o repositório
git clone https://github.com/SEU-USUARIO/brain-tumor-mri-pytorch.git
cd brain-tumor-mri-pytorch

#### 3. Criar e ativar um ambiente virtual
* Linux / macOS:
  python3 -m venv venv
  source venv/bin/activate
* Windows (PowerShell ou Prompt):
  python -m venv venv
  .\venv\Scripts\activate

#### 4. Instalar as dependências
pip install -r requirements.txt

#### 5. Organizar os dados
Coloque as imagens dentro de uma pasta (ex: data/Training) dividida pelas 4 categorias:
data/Training/
├── glioma_tumor/
├── meningioma_tumor/
├── no_tumor/
└── pituitary_tumor/

#### 6. Executar o Treinamento
python -m src.train --data_path "data/Training" --epochs 40 --batch_size 32 --lr 0.001 --img_size 128 --save_dir "checkpoints"

#### 7. Testar Inferência em uma Imagem Nova
Para classificar um exame individual:
python -m src.predict --image_path "data/Training/glioma_tumor/sample.jpg" --checkpoint "checkpoints/best_model.pth"

Exemplo de saída no console:
--- Resultado da Inferência ---
Diagnóstico: glioma_tumor
Confiança  : 97.85%

Distribuição de Probabilidades:
  glioma_tumor        : 97.85%
  meningioma_tumor    :  1.20%
  no_tumor            :  0.45%
  pituitary_tumor     :  0.50%

---

## 📊 Pipeline de Transformações (Data Augmentation)

Para evitar overfitting e elevar a capacidade de generalização da CNN, as imagens passam pelas seguintes operações antes do treino:
1. **Redimensionamento:** Interpolação para 128 x 128 pixels
2. **Espelhamento Horizontal:** Flip randômico (p=0.5)
3. **Rotação:** Variação de ± 15°
4. **Normalização:** Padrão ImageNet (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])

---

## 📜 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais detalhes.
