import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    def reverse_step(self, x, t):
        correction = (0.01 * (t+1)) * torch.tanh(x)
        return x - correction

class ClassifierModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassifierModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def joint_loss_fn(x_clean, x_reconstructed, classifier_logits, true_labels, lambda_c=1.0):
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x_clean)
    classification_loss = nn.CrossEntropyLoss()(classifier_logits, true_labels)
    return reconstruction_loss + lambda_c * classification_loss

def adaptive_reverse_diffusion(original_image, diffusion_model, classifier, confidence_threshold=0.9, max_steps=50):
    """
    Runs reverse diffusion adaptively until classifier confidence exceeds threshold.
    Returns final purified image, number of steps taken, and a list of intermediate confidence scores.
    """
    purified_image = original_image.clone()
    confidence_history = []
    step = 0
    for step in range(max_steps):
        purified_image = diffusion_model.reverse_step(purified_image, step)
        with torch.no_grad():
            logits = classifier(purified_image.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            max_confidence = probs.max().item()
            confidence_history.append(max_confidence)
        print(f"Adaptive Diffusion - Step {step+1}: Confidence = {max_confidence:.4f}")
        if max_confidence > confidence_threshold:
            break
    return purified_image, step+1, confidence_history

def compute_psnr(original, reconstructed):
    mse = ((original - reconstructed) ** 2).mean().item()
    if mse == 0:
        return 100
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr
