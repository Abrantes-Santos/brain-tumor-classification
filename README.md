# Brain-tumor-classification
Brain MRI tumor classification (Glioma, Meningioma, Pituitary, No Tumor) using PyTorch and CNNs


# Brain MRI Tumor Classification with Deep Learning (PyTorch)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pipeline end-to-end em PyTorch para detecção e classificação multi-classe de tumores cerebrais a partir de exames de Ressonância Magnética (MRI). O modelo categoriza os exames em 4 classes clínicas com alta precisão e suporte a inferência via CLI.

---

## 🎯 Classes Diagnosticadas

* **Glioma (`glioma_tumor`)**
* **Meningioma (`meningioma_tumor`)**
* **Adenoma Pituitário (`pituitary_tumor`)**
* **Controle Saudável (`no_tumor`)**

---

## 🏗️ Arquitetura da Rede (`BrainTumorCNN`)

A arquitetura foi projetada para processar tensores RGB normalizados de dimensão $(3, 128, 128)$, utilizando blocos convolucionais com Batch Normalization para acelerar a convergência e regularização via Dropout na camada densa:

```text
Entrada: (3, 128, 128)
│
├── [Bloco Conv 1]: Conv2d(3 -> 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)  => (32, 64, 64)
├── [Bloco Conv 2]: Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) => (64, 32, 32)
├── [Bloco Conv 3]: Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) => (128, 16, 16)
│
├── [Classificador]: Flatten() => Vetor de 32.768 dimensões
├── Linear(32.768 -> 256) -> ReLU -> Dropout(p=0.4)
└── Linear(256 -> 4) => Logits de Saída
