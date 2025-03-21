import argparse
from pathlib import Path
import torch
import torch.nn as nn

from model import MNIST_CNN
from data import get_testLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def find_checkpoint(dir_path: Path, name: str = None):
    if name:
        path = dir_path / name
        if not path.exists():
            raise FileNotFoundError(f"No such checkpoint: {path}")
        return path

    ckpts = sorted(dir_path.glob("mnist_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {dir_path}")
    return ckpts[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Filename of checkpoint in checkpoints/ folder")
    args = parser.parse_args()

    test_loader = get_testLoader()
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_path = find_checkpoint(ckpt_dir, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path.name}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
