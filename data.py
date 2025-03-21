from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Pre-processing Transforms
## first we want to transform our PIL raw image to a tensor
## second we want to normalize and shift our distribution for a faster covergance.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

# create function to return train and test loaders

def get_trainLoader(batchSize=64):
    mnist_train = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform)

    train_loader = DataLoader(
    mnist_train, 
    batch_size=batchSize, 
    shuffle=True)

    print('---Train Loader Generated---')

    return train_loader

def get_testLoader(batchSize=64):
    mnist_test = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform)

    test_loader = DataLoader(
    mnist_test, 
    batch_size=batchSize, 
    shuffle=False)

    print('---Test Loader Generated---')

    return test_loader