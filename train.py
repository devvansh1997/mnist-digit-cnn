## Imports
from model import MNIST_CNN
from data import get_trainLoader, get_testLoader

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure determinism (may slow down GPU training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)

## Global Constants
# before we begin training
# need to figure out what device we are on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Function Defitions
def save_checkpoint(model, optimizer, epoch, filepath="checkpoint.pth"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, filepath)

def load_checkpoint(model, optimizer, filepath="checkpoint.pth"):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]

def train_model(model, loader, optimizer, criterion):

    # put model in to training mode
    model.train()

    # training loop
    for images, labels in loader:

        # put the data on the device
        images, labels = images.to(device), labels.to(device)

        # zero out gradients
        optimizer.zero_grad()

        # calculate loss
        logits = model(images)
        loss = criterion(logits, labels)

        # calculate gradients based on the loss
        loss.backward()

        # update model params
        optimizer.step()

def validate_model(model, test_loader, criterion, epoch):
    
    # put model in to eval mode
    model.eval()

    # setup loss variables
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:

            # put the data on the device
            images, labels = images.to(device), labels.to(device)

            # calculate loss
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)

            # predictions
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
    
    # compute average loss & accuracy
    val_loss /= len(test_loader.dataset)
    val_acc = correct / len(test_loader.dataset)

    # print loss statistics
    print(f"Epoch {epoch + 1}: Validation Loss = {val_loss: .4f}, Accuracy = {val_acc: .4f}\n\n")

def main():
    # setup train and test loader
    train_loader = get_trainLoader()
    test_loader = get_testLoader()

    # initialize model from model.py
    model = MNIST_CNN().to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-03)

    # epoch range
    num_epochs = 5

    # training + validation loop
    for epoch in range(num_epochs):

        # train
        print(f"Epoch {epoch+1}: Training...")
        train_model(model, train_loader, optimizer, criterion)

        # validate
        print(f"Epoch {epoch+1}: Validating...")
        validate_model(model, test_loader, criterion, epoch)

        # save checkpoint
        save_checkpoint(model, optimizer, epoch, filepath=f"./checkpoints/mnist_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
