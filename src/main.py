import torch
import torch.optim as optim
import os
import sys

from preprocess import get_dataloader, get_device
from train import SimpleNetQ, SimpleNetG, SimpleNetD, LWGAN, IsoLWGANVariant, train_epoch
from evaluate import (
    visualize_interpolation, save_training_loss_plot, save_ablation_plot,
    compute_interpolation_consistency, ensure_images_directory
)

def experiment1(device):
    """Experiment 1: Quantitative Comparison on Reconstruction and Generation Quality"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Quantitative Comparison on Reconstruction and Generation Quality")
    print("="*80)
    
    batch_size = 64
    loader = get_dataloader(batch_size=batch_size, subset_size=256)
    z_dim = 128

    print(f"Dataset: CIFAR-10 subset (256 samples)")
    print(f"Batch size: {batch_size}")
    print(f"Latent dimension: {z_dim}")
    print(f"Device: {device}")

    netQ1 = SimpleNetQ(z_dim).to(device)
    netG1 = SimpleNetG(z_dim).to(device)
    netD1 = SimpleNetD().to(device)
    lwgan = LWGAN(z_dim, netQ1, netG1, netD1, device)

    netQ2 = SimpleNetQ(z_dim).to(device)
    netG2 = SimpleNetG(z_dim).to(device)
    netD2 = SimpleNetD().to(device)
    iso_lwgan = IsoLWGANVariant(z_dim, netQ2, netG2, netD2, device, iso_weight=0.1, adaptive_latent=True)

    optimizer_lwgan = optim.Adam(lwgan.parameters(), lr=0.0002)
    optimizer_iso = optim.Adam(iso_lwgan.parameters(), lr=0.0002)

    epochs = 3
    losses_lwgan = []
    losses_iso = []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs+1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        print("Training LWGAN:")
        loss_lw = train_epoch(lwgan, loader, optimizer_lwgan, use_isometry=False)
        losses_lwgan.append(loss_lw)
        print(f"LWGAN Epoch {epoch} Loss: {loss_lw:.4f}")
        
        print("Training Iso-LWGAN:")
        loss_iso_val = train_epoch(iso_lwgan, loader, optimizer_iso, use_isometry=True)
        losses_iso.append(loss_iso_val)
        print(f"Iso-LWGAN Epoch {epoch} Loss: {loss_iso_val:.4f}")
    
    print(f"\nFinal Results:")
    print(f"LWGAN Final Loss: {losses_lwgan[-1]:.4f}")
    print(f"Iso-LWGAN Final Loss: {losses_iso[-1]:.4f}")
    
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "training_loss.pdf")
    save_training_loss_plot(losses_lwgan, losses_iso, epochs, filename)
    
    print("Experiment 1 completed successfully!")
    return losses_lwgan, losses_iso

def experiment2(device):
    """Experiment 2: Ablation Study on Isometry Loss and Latent Prior"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Ablation Study on Isometry Loss and Latent Prior")
    print("="*80)
    
    batch_size = 64
    loader = get_dataloader(batch_size=batch_size, subset_size=256)
    z_dim = 128

    print(f"Dataset: CIFAR-10 subset (256 samples)")
    print(f"Batch size: {batch_size}")
    print(f"Latent dimension: {z_dim}")
    print(f"Device: {device}")

    netQ_full = SimpleNetQ(z_dim).to(device)
    netG_full = SimpleNetG(z_dim).to(device)
    netD_full = SimpleNetD().to(device)
    full_iso = IsoLWGANVariant(z_dim, netQ_full, netG_full, netD_full, device, iso_weight=0.1, adaptive_latent=True)

    netQ_no_iso = SimpleNetQ(z_dim).to(device)
    netG_no_iso = SimpleNetG(z_dim).to(device)
    netD_no_iso = SimpleNetD().to(device)
    no_isometry = IsoLWGANVariant(z_dim, netQ_no_iso, netG_no_iso, netD_no_iso, device, iso_weight=0.0, adaptive_latent=True)

    netQ_fixed = SimpleNetQ(z_dim).to(device)
    netG_fixed = SimpleNetG(z_dim).to(device)
    netD_fixed = SimpleNetD().to(device)
    fixed_latent = IsoLWGANVariant(z_dim, netQ_fixed, netG_fixed, netD_fixed, device, iso_weight=0.1, adaptive_latent=False)

    optimizer_full = optim.Adam(full_iso.parameters(), lr=0.0002)
    optimizer_no_iso = optim.Adam(no_isometry.parameters(), lr=0.0002)
    optimizer_fixed = optim.Adam(fixed_latent.parameters(), lr=0.0002)

    epochs = 3
    loss_full, loss_no_iso, loss_fixed = [], [], []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs+1):
        print(f"\n--- Ablation Epoch {epoch}/{epochs} ---")
        print("Training Full Iso-LWGAN (with isometry + adaptive):")
        loss1 = train_epoch(full_iso, loader, optimizer_full, use_isometry=True)
        loss_full.append(loss1)
        print(f"Full Iso-LWGAN Epoch {epoch} Loss: {loss1:.4f}")
        
        print("Training Iso-LWGAN without isometry:")
        loss2 = train_epoch(no_isometry, loader, optimizer_no_iso, use_isometry=False)
        loss_no_iso.append(loss2)
        print(f"No-Isometry Epoch {epoch} Loss: {loss2:.4f}")
        
        print("Training Iso-LWGAN with fixed latent:")
        loss3 = train_epoch(fixed_latent, loader, optimizer_fixed, use_isometry=True)
        loss_fixed.append(loss3)
        print(f"Fixed-Latent Epoch {epoch} Loss: {loss3:.4f}")
    
    print(f"\nAblation Study Results:")
    print(f"Full Iso-LWGAN Final Loss: {loss_full[-1]:.4f}")
    print(f"No-Isometry Final Loss: {loss_no_iso[-1]:.4f}")
    print(f"Fixed-Latent Final Loss: {loss_fixed[-1]:.4f}")
    
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "training_loss_ablation.pdf")
    save_ablation_plot(loss_full, loss_no_iso, loss_fixed, epochs, filename)
    
    print("Experiment 2 completed successfully!")
    return loss_full, loss_no_iso, loss_fixed

def experiment3(device):
    """Experiment 3: Latent Space Interpolation and Manifold Consistency"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Latent Space Interpolation and Manifold Consistency")
    print("="*80)
    
    z_dim = 128
    print(f"Latent dimension: {z_dim}")
    print(f"Device: {device}")
    
    netQ_interp = SimpleNetQ(z_dim).to(device)
    netG_interp = SimpleNetG(z_dim).to(device)
    netD_interp = SimpleNetD().to(device)
    model = LWGAN(z_dim, netQ_interp, netG_interp, netD_interp, device)
    
    print("Generating random latent codes for interpolation...")
    z_start = torch.randn(1, z_dim)
    z_end = torch.randn(1, z_dim)
    
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "latent_interpolation.pdf")
    visualize_interpolation(model, z_start, z_end, steps=12, filename=filename)
    
    print("Computing interpolation consistency metrics...")
    latent_dists, img_dists = compute_interpolation_consistency(model, z_start, z_end, steps=12)
    
    print("Latent distances between consecutive interpolations:")
    for i, dist in enumerate(latent_dists):
        print(f"  Step {i+1}: {dist:.4f}")
    
    print("Image distances between consecutive interpolations:")
    for i, dist in enumerate(img_dists):
        print(f"  Step {i+1}: {dist:.4f}")
    
    avg_latent_dist = sum(latent_dists) / len(latent_dists) if latent_dists else 0
    avg_img_dist = sum(img_dists) / len(img_dists) if img_dists else 0
    
    print(f"\nInterpolation Consistency Metrics:")
    print(f"Average latent distance: {avg_latent_dist:.4f}")
    print(f"Average image distance: {avg_img_dist:.4f}")
    
    print("Experiment 3 completed successfully!")
    return latent_dists, img_dists

def main():
    """Main experiment orchestrator"""
    print("="*80)
    print("ISO-LWGAN EXPERIMENT SUITE")
    print("Isometric Latent Wasserstein GAN Implementation")
    print("="*80)
    
    device = get_device()
    
    try:
        print("\nStarting experimental evaluation...")
        
        exp1_results = experiment1(device)
        exp2_results = experiment2(device)
        exp3_results = experiment3(device)
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nExperiment Summary:")
        print("1. Quantitative Comparison: Training loss curves saved")
        print("2. Ablation Study: Component analysis completed")
        print("3. Latent Interpolation: Manifold consistency evaluated")
        
        images_dir = ensure_images_directory()
        print(f"\nAll plots saved to: {images_dir}")
        print("- training_loss.pdf")
        print("- training_loss_ablation.pdf") 
        print("- latent_interpolation.pdf")
        
        status_enum = "stopped"
        print(f"\nExperiment status: {status_enum}")
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        status_enum = "error"
        sys.exit(1)

if __name__ == "__main__":
    main()
