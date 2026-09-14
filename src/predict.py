import argparse
import os
from typing import Dict, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import get_transforms
from src.model import BrainTumorCNN
from src.utils import get_device


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[BrainTumorCNN, list]:
    """Carrega os pesos do modelo e a lista de classes salvas no checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERRO] Checkpoint não encontrado: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", [
        "glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"
    ])

    model = BrainTumorCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(
    image_path: str,
    model: BrainTumorCNN,
    class_names: list,
    device: torch.device,
    img_size: int = 128
) -> Dict[str, any]:
    """
    Executa inferência em uma imagem única a partir do seu caminho em disco.
    Retorna a classe predita, a confiança (%) e as probabilidades por classe.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERRO] Imagem não encontrada: {image_path}")

    # Carrega e converte para RGB (caso a ressonância seja monocromática ou RGBA)
    raw_img = Image.open(image_path).convert("RGB")
    _, eval_transform = get_transforms(img_size=img_size)
    tensor_img = eval_transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        top_prob, top_idx = torch.max(probabilities, dim=0)

    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item() * 100.0

    all_probs = {
        cls: round(prob.item() * 100.0, 2)
        for cls, prob in zip(class_names, probabilities)
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": all_probs
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência com BrainTumorCNN")
    parser.add_argument("--image_path", type=str, required=True, help="Caminho da imagem de MRI para teste")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Caminho do arquivo .pth")
    parser.add_argument("--img_size", type=int, default=128, help="Resolução esperada pelo modelo")

    args = parser.parse_args()
    device = get_device()

    model, class_names = load_checkpoint(args.checkpoint, device)
    result = predict_image(args.image_path, model, class_names, device, args.img_size)

    print("\n--- Resultado da Inferência ---")
    print(f"Diagnóstico: {result['predicted_class']}")
    print(f"Confiança  : {result['confidence']}%")
    print("\nDistribuição de Probabilidades:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:<20}: {prob:>5.2f}%")
