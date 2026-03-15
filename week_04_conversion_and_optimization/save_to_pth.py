import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50


def build_model() -> torch.nn.Module:
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 128)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Save PyTorch model to .pth")
    parser.add_argument(
        "--output",
        type=str,
        default="model.pth",
        help="Output .pth file path",
    )
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help="Save full model object instead of state_dict",
    )
    args = parser.parse_args()

    model = build_model()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_full_model:
        torch.save(model, str(output_path))
    else:
        torch.save(model.state_dict(), str(output_path))

    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
