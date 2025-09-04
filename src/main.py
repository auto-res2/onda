import os
import numpy as np
import torch

from preprocess import get_device, set_random_seeds, get_trexfit_dataloader
from train import (
    FiTBaseline, TRExFiT, TRExFiTAdaptivePE,
    train_model, train_with_consistency
)
from evaluate import ensure_images_directory, plot_loss_curve, visualize_token_grid

def experiment_1_resolution_extrapolation(device):
    print("="*80)
    print("EXPERIMENT 1: Trainable Resolution Extrapolation vs Baseline")
    print("="*80)
    dataloader = get_trexfit_dataloader(batch_size=8, num_samples=32, image_size=(256,256))
    baseline = FiTBaseline(in_channels=3, hidden_size=64)
    trexfit = TRExFiT(in_channels=3, hidden_size=64)
    baseline, hist_base = train_model(baseline, dataloader, epochs=1, device=device)
    trexfit, hist_trex = train_model(trexfit, dataloader, epochs=1, device=device)
    images_dir = ensure_images_directory()
    plot_loss_curve(hist_base, "Baseline Training Loss", os.path.join(images_dir, "training_loss_baseline.pdf"))
    plot_loss_curve(hist_trex, "TRExFiT Training Loss", os.path.join(images_dir, "training_loss_trexfit.pdf"))
    imgs, _ = next(iter(dataloader))
    target_res = (32, 32)
    _ = baseline(imgs.to(device), target_resolution=target_res)
    _ = trexfit(imgs.to(device), target_resolution=target_res)
    print("Experiment 1 complete.\n")

def experiment_2_multi_scale_consistency(device):
    print("="*80)
    print("EXPERIMENT 2: Multi-Scale Consistency Loss")
    print("="*80)
    dataloader = get_trexfit_dataloader(batch_size=8, num_samples=32, image_size=(256,256))
    model_with = TRExFiT(in_channels=3, hidden_size=64)
    model_without = TRExFiT(in_channels=3, hidden_size=64)
    model_with, hist_with = train_with_consistency(model_with, dataloader, consistency_weight=0.1, epochs=1, device=device)
    model_without, hist_without = train_model(model_without, dataloader, epochs=1, device=device)
    images_dir = ensure_images_directory()
    plot_loss_curve(hist_with, "TRExFiT + Consistency", os.path.join(images_dir, "training_loss_consistency.pdf"))
    plot_loss_curve(hist_without, "TRExFiT Standard", os.path.join(images_dir, "training_loss_standard.pdf"))
    print("Experiment 2 complete.\n")

def experiment_3_adaptive_positional_encoding(device):
    print("="*80)
    print("EXPERIMENT 3: Adaptive Positional Encoding")
    print("="*80)
    dataloader = get_trexfit_dataloader(batch_size=8, num_samples=16, image_size=(256,256))
    adaptive = TRExFiTAdaptivePE(in_channels=3, hidden_size=64, d_model=64, use_adaptive_pe=True)
    standard = TRExFiTAdaptivePE(in_channels=3, hidden_size=64, d_model=64, use_adaptive_pe=False)
    imgs, _ = next(iter(dataloader))
    out_ad = adaptive(imgs)
    out_std = standard(imgs)
    out_img = out_ad[0].detach().cpu().numpy().transpose(1,2,0)
    images_dir = ensure_images_directory()
    visualize_token_grid(out_img, token_resolution=(8,8), filename=os.path.join(images_dir, "inference_token_grid.pdf"))
    print("Experiment 3 complete.\n")

def main():
    print("="*80)
    print("TRExFiT EXPERIMENT SUITE")
    print("Trainable Resolution Extrapolation with Consistency + Adaptive PE")
    print("="*80)
    set_random_seeds(42)
    device = get_device()
    try:
        experiment_1_resolution_extrapolation(device)
        experiment_2_multi_scale_consistency(device)
        experiment_3_adaptive_positional_encoding(device)
        print("="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        images_dir = ensure_images_directory()
        print(f"All plots saved to: {images_dir}")
        print("- training_loss_baseline.pdf")
        print("- training_loss_trexfit.pdf")
        print("- training_loss_consistency.pdf")
        print("- training_loss_standard.pdf")
        print("- inference_token_grid.pdf")
        status_enum = "stopped"
        print(f"\nExperiment status: {status_enum}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        status_enum = "error"
        raise

if __name__ == "__main__":
    main()
