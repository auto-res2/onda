import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

def get_environment():
    """
    Load CIFAR-10 test dataset for adversarial purification experiments.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    return test_loader

def generate_expert_data(test_loader, num_samples=100):
    """
    Generate sample data from CIFAR-10 test set.
    Returns a batch of images and labels for experiments.
    """
    batch_idx, (images, labels) = next(enumerate(test_loader))
    return images, labels

def get_device():
    """Get the appropriate device (GPU if available, CPU otherwise)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU")
    return device

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
