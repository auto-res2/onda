import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils
import os

def interpolate_latent(z_start, z_end, steps=10):
    """Interpolate between two latent codes"""
    ratios = np.linspace(0, 1, steps)
    interpolated = [(1 - r) * z_start + r * z_end for r in ratios]
    return torch.stack(interpolated, dim=0)

def visualize_interpolation(model, z_start, z_end, steps=10, filename="latent_interpolation.pdf"):
    """Visualize latent space interpolation and save as PDF"""
    model.netG.eval()
    with torch.no_grad():
        interpolated_z = interpolate_latent(z_start, z_end, steps=steps)
        interpolated_z = interpolated_z.to(model.device)
        generated_imgs = model.netG(interpolated_z)
    generated_imgs = generated_imgs.cpu()
    grid = utils.make_grid(generated_imgs, nrow=steps, normalize=True)
    plt.figure(figsize=(15, 3))
    plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
    plt.title('Latent Space Interpolation')
    plt.axis('off')
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved latent interpolation plot as: {filename}")

def save_training_loss_plot(losses_lwgan, losses_iso, epochs, filename="training_loss.pdf"):
    """Save training loss comparison plot as PDF"""
    plt.figure()
    epochs_range = range(1, epochs+1)
    plt.plot(epochs_range, losses_lwgan, label='LWGAN')
    plt.plot(epochs_range, losses_iso, label='Iso-LWGAN')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")

def save_ablation_plot(loss_full, loss_no_iso, loss_fixed, epochs, filename="training_loss_ablation.pdf"):
    """Save ablation study plot as PDF"""
    plt.figure()
    epochs_range = range(1, epochs+1)
    plt.plot(epochs_range, loss_full, label='Full Iso (adaptive)')
    plt.plot(epochs_range, loss_no_iso, label='No Isometry Loss')
    plt.plot(epochs_range, loss_fixed, label='Fixed Latent Prior')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Ablation Study: Training Loss Comparison")
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")

def compute_interpolation_consistency(model, z_start, z_end, steps=12):
    """Compute quantitative measures of interpolation consistency"""
    interpolated_z = interpolate_latent(z_start, z_end, steps=steps)
    latent_dists = []
    img_dists = []
    previous_z = None
    previous_img = None
    model.netG.eval()
    with torch.no_grad():
        for z in interpolated_z:
            z = z.unsqueeze(0).to(model.device)
            img = model.netG(z)
            if previous_z is not None:
                latent_dists.append(torch.linalg.norm(previous_z - z).item())
                img_dists.append(torch.linalg.norm(previous_img - img).item())
            previous_z = z
            previous_img = img
    return latent_dists, img_dists

def ensure_images_directory():
    """Ensure the images directory exists"""
    images_dir = ".research/iteration1/images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir
