import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from torchvision.models import resnet18
from torch.nn import Sequential
import numpy as np
import time

def get_classifier():
    """
    Load a ResNet-18 model for CIFAR-10 classification.
    Note: In a real experiment you would load a properly pretrained classifier on CIFAR-10.
    Here, for demonstration purposes, we use random weights.
    """
    classifier = resnet18(num_classes=10)
    classifier.eval()
    
    for param in classifier.parameters():
        param.requires_grad = False
    
    return classifier

def get_feature_extractor():
    """
    Use a pretrained ResNet-18 as a feature extractor: remove final fully-connected layer.
    """
    feature_extractor = resnet18(pretrained=True)
    modules = list(feature_extractor.children())[:-1]
    feature_extractor = Sequential(*modules)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    return feature_extractor

def purify_purifypp(image_adv):
    """
    Purification using the baseline Purify++ method.
    For demonstration we implement a dummy reverse diffusion that returns a slightly smoothed image.
    In practice, this would run a reverse diffusion with fixed parameters.
    """
    with torch.no_grad():
        purified = image_adv * 0.9 + torch.randn_like(image_adv) * 0.05
    return purified

def purify_acdp(image_adv):
    """
    Purification using the proposed Adaptive Classifier-guided Diffusion Purification (ACDP).
    For demonstration this placeholder uses a similar reverse diffusion but with a dynamic component.
    """
    with torch.no_grad():
        scaling = torch.clamp(torch.abs(image_adv).mean(dim=[1,2,3], keepdim=True), 0.8, 1.2)
        purified = image_adv * 0.92 + torch.randn_like(image_adv) * 0.05 * scaling
    return purified

def adaptive_purification_with_logging(image_adv, classifier, num_steps=50):
    """
    This function implements a dummy adaptive purification routine that logs the evolution
    of the guidance weight (λ), noise scaling, and classifier confidence over 'num_steps' steps.
    """
    guidance_log = []
    noise_log = []
    confidence_log = []
    
    purified = image_adv.clone()
    
    for step in range(num_steps):
        with torch.no_grad():
            outputs = classifier(purified)
            probs = F.softmax(outputs, dim=1)
            confidences = probs.max(dim=1).values
            
            lambda_val = torch.where(confidences < 0.7, torch.tensor(1.0, device=purified.device),
                                     torch.tensor(0.5, device=purified.device))
            noise_sigma = lambda_val * 0.1
            
            guidance_log.append(lambda_val.mean().item())
            noise_log.append(noise_sigma.mean().item())
            confidence_log.append(confidences.mean().item())
            
            lambda_tensor = lambda_val.view(-1,1,1,1)
            noise = torch.randn_like(purified) * noise_sigma.view(-1,1,1,1)
            purified = purified - lambda_tensor * (purified - image_adv) + noise
            
    return purified, guidance_log, noise_log, confidence_log
