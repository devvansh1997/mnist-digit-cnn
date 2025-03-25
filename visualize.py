import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

from model import MNIST_CNN
from data import get_trainLoader, get_testLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS_DIR = Path(__file__).parent / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

def dataset_stats(loader):
    labels = loader.dataset.targets
    classes, counts = torch.unique(labels, return_counts=True)
    print(f"Number of classes: {len(classes)}")
    for cls, cnt in zip(classes.tolist(), counts.tolist()):
        print(f"  Class {cls}: {cnt} samples")
    print(f"Total training samples: {len(loader.dataset)}\n")

def show_examples(loader, classes_to_show=[0,1,2], samples_per_class=3):
    dataset = loader.dataset
    fig, axes = plt.subplots(len(classes_to_show), samples_per_class, figsize=(samples_per_class*2, len(classes_to_show)*2))
    for i, cls in enumerate(classes_to_show):
        idxs = (dataset.targets == cls).nonzero().flatten()[:samples_per_class]
        for j, idx in enumerate(idxs):
            axes[i,j].imshow(dataset[idx][0].squeeze(), cmap="gray")
            axes[i,j].axis("off")
            if j == 0:
                axes[i,j].set_title(f"Class {cls}")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "sample_images.png")
    plt.close(fig)

def predict_and_display(model, loader, n=9):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:n].to(device), labels[:n].to(device)
    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    fig, axes = plt.subplots(3, 3, figsize=(6,6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"True: {labels[i].item()} → Pred: {preds[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "sample_predictions.png")
    plt.close(fig)

def plot_class_distribution(loader):
    labels = loader.dataset.targets.numpy()
    classes, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(classes, counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.set_title("Training Set Class Distribution")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "class_distribution.png")
    plt.close(fig)


def plot_confusion(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            all_preds.append(model(images).argmax(dim=1).cpu())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "confusion_matrix.png")
    plt.close(fig)

if __name__ == "__main__":
    train_loader = get_trainLoader()
    test_loader = get_testLoader()

    print("\n--- Dataset Statistics ---")
    dataset_stats(train_loader)

    print("--- Saving Class Distribution ---")
    plot_class_distribution(train_loader)

    print("--- Saving Sample Images ---")
    show_examples(train_loader)

    model = MNIST_CNN().to(device)
    ckpt_dir = Path(__file__).parent / "checkpoints"
    latest = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)[-1]
    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    print(f"\nLoaded checkpoint: {latest.name}")

    print("--- Saving Sample Predictions ---")
    predict_and_display(model, test_loader)

    print("--- Saving Confusion Matrix ---")
    plot_confusion(model, test_loader)

    print(f"\nAll visualizations saved to `{VIS_DIR}`")
