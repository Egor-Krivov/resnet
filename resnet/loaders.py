import torch

import torchvision
import torchvision.transforms as transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


def get_train_loader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root='~/cifar-10/', train=True, download=True, transform=transform_train)
    return torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)


def get_test_loader():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(root='~/cifar-10/', train=False, download=True, transform=transform_test)
    return torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
