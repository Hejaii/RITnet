import argparse
import os

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from dataset import transform
from models import model_dict
from utils import get_predictions


def preprocess_image(path: str) -> torch.Tensor:
    """Load and preprocess an image for RITnet.

    This function mirrors the preprocessing used during training/test.
    The input image is converted to grayscale, gamma corrected, and
    contrast limited adaptive histogram equalization (CLAHE) is applied
    before normalizing to a tensor.
    """
    pilimg = Image.open(path).convert("L")

    # Gamma correction with factor 0.8
    table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    pilimg = cv2.LUT(np.array(pilimg), table)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(np.array(np.uint8(pilimg)))
    img = Image.fromarray(img)

    # Normalize to tensor
    img = transform(img)
    return img.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a single eye image using RITnet.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", default="segmentation.png", help="Path to save the predicted mask")
    parser.add_argument("--weights", default="best_model.pkl", help="Path to model weights")
    parser.add_argument("--model", default="densenet", choices=list(model_dict.keys()), help="Model architecture")
    parser.add_argument("--useGPU", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.useGPU and torch.cuda.is_available() else "cpu")
    if args.useGPU and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
    print(f"Using device: {device}")

    net = model_dict[args.model].to(device)
    state = torch.load(args.weights, map_location=device)
    net.load_state_dict(state)
    net.eval()

    img = preprocess_image(args.image).to(device)

    with torch.no_grad():
        pred = net(img)
        mask = get_predictions(pred)[0].cpu().numpy()

    plt.imsave(args.output, mask)
    print(f"Saved segmentation mask to {args.output}")


if __name__ == "__main__":
    main()
