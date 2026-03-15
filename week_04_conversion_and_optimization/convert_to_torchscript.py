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
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TorchScript")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to .pt/.pth state_dict checkpoint (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.ts",
        help="Output TorchScript file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for tracing",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=device)
    scripted_model = torch.jit.trace(model, dummy_input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted_model.save(str(output_path))
    print(f"TorchScript model saved to: {output_path}")


if __name__ == "__main__":
    main()
