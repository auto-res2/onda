import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def isometry_loss(z_batch, img_batch, netG, distance_metric=torch.nn.functional.mse_loss):
    """
    Compute an isometry loss that penalizes the difference
    between the distance between latent codes and the distance 
    between the corresponding generated images.
    """
    n = z_batch.size(0)
    latent_loss = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            latent_dist = torch.linalg.norm(z_batch[i] - z_batch[j])
            img_i = netG(z_batch[i].unsqueeze(0))
            img_j = netG(z_batch[j].unsqueeze(0))
            img_dist = torch.linalg.norm(img_i - img_j)
            latent_loss += torch.abs(latent_dist - img_dist)
            count += 1
    if count > 0:
        latent_loss = latent_loss / count
    return latent_loss

def _gradient_penalty(real, fake, critic):
    """Simple L2 penalty on difference between gradients (dummy implementation)"""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = critic(interpolates, rank=0)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def mmd_penalty(z_hat, z, kernel="IMQ", sigma2_p=1.0):
    """Dummy MMD penalty"""
    return torch.abs(z_hat.mean() - z.mean())

def gen_noise_with_rank(n, z_dim, rank, device):
    """For simplicity, ignore rank and return standard normal noise."""
    return torch.randn(n, z_dim, device=device)

class SimpleNetQ(nn.Module):
    """A simple encoder mapping images to latent codes."""
    def __init__(self, z_dim):
        super(SimpleNetQ, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(32*8*8, z_dim)
    
    def forward(self, x, rank=0):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        z = self.fc(x)
        return z

class SimpleNetG(nn.Module):
    """A simple generator mapping latent codes to images."""
    def __init__(self, z_dim):
        super(SimpleNetG, self).__init__()
        self.fc = nn.Linear(z_dim, 32*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, 32, 8, 8)
        x = self.deconv(x)
        return x

class SimpleNetD(nn.Module):
    """A simple discriminator."""
    def __init__(self):
        super(SimpleNetD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(32*8*8, 1)
    
    def forward(self, x, rank=0):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        out = self.fc(x)
        return out

class LWGAN(nn.Module):
    def __init__(self, z_dim, netQ, netG, netD, device=torch.device("cpu")):
        super(LWGAN, self).__init__()
        self.z_dim = z_dim
        self.netQ = netQ
        self.netG = netG
        self.netD = netD
        self.device = device

    @torch.jit.export
    def D_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        post_data = self.netG(self.netQ(real_data, rank))
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = -torch.abs(diff) if abs else -diff
        return losses.mean()

    @torch.jit.export
    def GQ_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = l2 + torch.abs(diff) if abs else l2 + diff
        return losses.mean()

    @torch.jit.ignore
    def gradient_penalty_D(self, x, z, rank: int):
        x_hat = self.netG(z)
        x_tilde = self.netG(self.netQ(x, rank))
        return _gradient_penalty(x_hat, x_tilde, lambda x: self.netD(x, rank))

    @torch.jit.export
    def mmd_penalty(self, real_data, rank: int, lambda_mmd: float):
        n = real_data.shape[0]
        mmd = torch.tensor([0.0], device=real_data.device)
        if lambda_mmd != 0.0:
            z = gen_noise_with_rank(n, self.z_dim, rank, self.device)
            z_hat = self.netQ(real_data, rank)
            mmd = lambda_mmd * mmd_penalty(z_hat, z, kernel="IMQ", sigma2_p=1.0)
        return mmd

    @torch.jit.ignore
    def dual_loss(self, x1, x2, rank: int, lambda_gp: float):
        n = x1.shape[0]
        noise = gen_noise_with_rank(n, self.z_dim, rank, self.device)
        fake_data = self.netG(noise)
        cost_D = self.D_loss(x1, fake_data, rank, abs=False)
        z = gen_noise_with_rank(n, self.z_dim, rank, self.device)
        gp_D = self.gradient_penalty_D(x2.data, z.data, rank)
        dual_cost = cost_D + lambda_gp * gp_D
        return dual_cost

    @torch.jit.export
    def recon_loss(self, real_data, rank: int):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        return l2.mean()

    @torch.jit.export
    def forward(self, x1, x2, rank: int, lambda_mmd: float, lambda_rank: float):
        n = x1.shape[0]
        noise = gen_noise_with_rank(n, self.z_dim, rank, self.device)
        fake_data = self.netG(noise)
        cost_GQ = self.GQ_loss(x1, fake_data, rank, abs=False)
        mmd = self.mmd_penalty(x2, rank, lambda_mmd)
        primal_cost = cost_GQ + mmd + lambda_rank * rank
        return primal_cost

class IsoLWGANVariant(LWGAN):
    def __init__(self, z_dim, netQ, netG, netD, device, iso_weight=0.1, adaptive_latent=True):
        super(IsoLWGANVariant, self).__init__(z_dim, netQ, netG, netD, device)
        self.iso_weight = iso_weight
        self.adaptive_latent = adaptive_latent
        if self.adaptive_latent:
            self.latent_transform = nn.Linear(z_dim, z_dim)
        else:
            self.latent_transform = nn.Identity()

    def forward(self, x1, x2, rank: int, lambda_mmd: float, lambda_rank: float):
        basic_loss = super().forward(x1, x2, rank, lambda_mmd, lambda_rank)
        latent_codes = self.netQ(x1, rank)
        latent_codes = self.latent_transform(latent_codes)
        iso_loss = isometry_loss(latent_codes, x1, self.netG)
        return basic_loss + self.iso_weight * iso_loss

def train_epoch(model, loader, optimizer, use_isometry=False, lambda_mmd=0.1, lambda_rank=0.0):
    """Training loop for one epoch"""
    model.train()
    epoch_loss = 0.0
    for batch_idx, (real_images, _) in enumerate(loader):
        real_images = real_images.to(model.device)
        rank = torch.randint(low=0, high=5, size=(1,)).item()  
        noise = torch.randn(real_images.size(0), model.z_dim, device=model.device)
        fake_images = model.netG(noise)
        optimizer.zero_grad()
        loss_primal = model.forward(real_images, real_images, rank, lambda_mmd, lambda_rank)
        if use_isometry:
            latent_codes = model.netQ(real_images, rank)
            loss_iso = isometry_loss(latent_codes, real_images, model.netG)
            loss = loss_primal + 0.1 * loss_iso
        else:
            loss = loss_primal
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    avg_loss = epoch_loss / len(loader)
    return avg_loss
