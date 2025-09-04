import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
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

class DummyBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.conv(x))

class DummyDecoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, latent):
        return torch.sigmoid(self.conv(latent))

class FiTBaseline(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.backbone = DummyBackbone(in_channels, hidden_size)
        self.decoder = DummyDecoder(hidden_size, in_channels)
    def forward(self, x, target_resolution=None):
        latent = self.backbone(x)
        if target_resolution is not None:
            latent = F.interpolate(latent, size=target_resolution, mode='bicubic', align_corners=False)
        return self.decoder(latent)

class TrainableUpsampler(nn.Module):
    def __init__(self, in_channels, up_factor=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (up_factor ** 2), 3, padding=1),
            nn.PixelShuffle(up_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    def forward(self, x): 
        return self.net(x)

class TRExFiT(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.backbone = DummyBackbone(in_channels, hidden_size)
        self.trainable_upsampler = TrainableUpsampler(hidden_size, up_factor=2)
        self.decoder = DummyDecoder(hidden_size, in_channels)
    def forward(self, x, target_resolution=None):
        latent = self.backbone(x)
        if target_resolution is not None:
            latent = self.trainable_upsampler(latent)
            h, w = latent.shape[2:]
            H, W = target_resolution
            if (h, w) != (H, W):
                latent = F.interpolate(latent, size=target_resolution, mode='bicubic', align_corners=False)
        return self.decoder(latent)

def multi_scale_loss(latent_original, latent_scaled):
    B, C, H, W = latent_original.shape
    if latent_scaled.shape != latent_original.shape:
        latent_scaled = F.interpolate(latent_scaled, size=(H, W), mode='bilinear', align_corners=False)
    
    lo = latent_original.view(B, C, -1)
    ls = latent_scaled.view(B, C, -1)
    lo = lo / (lo.norm(dim=1, keepdim=True) + 1e-8)
    ls = ls / (ls.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (lo * ls).sum(dim=1).mean()
    return 1 - cos_sim

def train_model(model, dataloader, epochs=1, lr=1e-4, device='cpu'):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, imgs)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / max(1, len(dataloader)))
    return model, history

def train_with_consistency(model, dataloader, consistency_weight=0.1, epochs=1, lr=1e-4, device='cpu'):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base_loss_fn = nn.MSELoss()
    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            opt.zero_grad()
            latent_orig = model.backbone(imgs)
            out = model.decoder(latent_orig)
            scaled = F.interpolate(imgs, scale_factor=0.8, mode='bilinear', align_corners=False)
            latent_scaled = model.backbone(scaled)
            loss = base_loss_fn(out, imgs) + consistency_weight * multi_scale_loss(latent_orig, latent_scaled)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / max(1, len(dataloader)))
    return model, history

class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_resolution=(256,256)):
        super().__init__()
        self.d_model = d_model
        self.adapt_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv2d(2, d_model, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        y = torch.linspace(0, 1, steps=H, device=x.device)
        z = torch.linspace(0, 1, steps=W, device=x.device)
        gy, gz = torch.meshgrid(y, z, indexing='ij')
        pe = torch.stack([gy, gz], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        pe = self.proj(pe) * self.adapt_scale
        return x + pe

class TRExFiTAdaptivePE(TRExFiT):
    def __init__(self, in_channels=3, hidden_size=64, d_model=64, use_adaptive_pe=True):
        super().__init__(in_channels=in_channels, hidden_size=hidden_size)
        self.use_adaptive_pe = use_adaptive_pe
        self.adaptive_pe = AdaptivePositionalEncoding(d_model=d_model) if use_adaptive_pe else nn.Identity()
    def forward(self, x, target_resolution=None):
        latent = self.backbone(x)
        latent = self.adaptive_pe(latent) if self.use_adaptive_pe else latent
        if target_resolution is not None:
            latent = self.trainable_upsampler(latent)
            latent = F.interpolate(latent, size=target_resolution, mode='bicubic', align_corners=False)
        return self.decoder(latent)
