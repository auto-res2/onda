import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloader(batch_size=64, subset_size=256, train=True):
    """
    Load CIFAR-10 dataset with proper transforms for the experiments.
    
    Args:
        batch_size: Batch size for data loader
        subset_size: Size of subset for quick testing (None for full dataset)
        train: Whether to load training or test set
    
    Returns:
        DataLoader for CIFAR-10 dataset
    """
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root='./data', 
        train=train, 
        transform=transform, 
        download=True
    )
    
    if subset_size:
        indices = list(range(subset_size))
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def get_device():
    """Get the appropriate device (GPU if available, CPU otherwise)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU")
    return device
