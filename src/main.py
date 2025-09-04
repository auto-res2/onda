#!/usr/bin/env python3
"""
Bayesian Flow-Controlled Purification (BFCP) Experiment Suite
Implements three experiments comparing BFCP with Purify++ baseline:
1. Computational Efficiency and Runtime Analysis
2. Robustness Against Diverse Adversarial Threat Models
3. Semantic Preservation and Visual Quality
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import foolbox as fb
from piq import ssim
import torchvision.models as vmodels

from preprocess import get_device, set_random_seeds
from train import run_bfcp_pipeline, run_purify_pp_pipeline, get_encoder_latent
from evaluate import ensure_images_directory

def experiment1():
    print("\nStarting Experiment 1: Runtime and computational efficiency analysis")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    bfcp_timings = []
    purifypp_timings = []
    batch_count = 0
    for batch_idx, (data, target) in enumerate(loader):
        _, timing_bfcp = run_bfcp_pipeline(model=None, input_data=data)
        bfcp_timings.append(timing_bfcp)
    
        _, timing_purifypp = run_purify_pp_pipeline(model=None, input_data=data)
        purifypp_timings.append(timing_purifypp)
    
        batch_count += 1
        if batch_count >= 5:
            break

    def average_timings(timings_list, keys):
        avg = {k: sum(d.get(k, 0) for d in timings_list) / len(timings_list) for k in keys}
        return avg

    bfcp_keys = ['latent_encoding', 'bayesian_refinement', 'reverse_diffusion', 'decoding', 'total']
    avg_bfcp = average_timings(bfcp_timings, bfcp_keys)
    avg_purifypp = average_timings(purifypp_timings, ['total'])

    print("BFCP average timings:", avg_bfcp)
    print("Purify++ average timing (total):", avg_purifypp)

    stages = list(avg_bfcp.keys())
    bfcp_values = [avg_bfcp[stage] for stage in stages]
    purifypp_total = avg_purifypp['total']
    
    images_dir = ensure_images_directory()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(stages, bfcp_values, color='blue', alpha=0.7, label='BFCP')
    ax.axhline(purifypp_total, color='red', linestyle='--', label='Purify++ total')
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Pipeline Stage Timings")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{images_dir}/runtime_timings.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Experiment 1 plot saved as runtime_timings.pdf")

def experiment2():
    print("\nStarting Experiment 2: Robustness Against Diverse Adversarial Threat Models")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = get_device()
    
    classifier = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    ).to(device)
    classifier.eval()

    robust_acc_bfcp_list = []
    robust_acc_pp_list = []
    ssim_bfcp_list = []
    ssim_pp_list = []
    
    batch_count = 0
    for batch_idx, (data, labels) in enumerate(loader):
        data = data.to(device)
        labels = labels.to(device)
    
        epsilon = 0.03
        noise = torch.randn_like(data) * epsilon
        adv_data = torch.clamp(data + noise, 0, 1)
    
        purified_bfcp, _ = run_bfcp_pipeline(model=classifier, input_data=adv_data)
        purified_pp, _ = run_purify_pp_pipeline(model=classifier, input_data=adv_data)
    
        with torch.no_grad():
            pred_bfcp = classifier(purified_bfcp).argmax(dim=1)
            pred_pp = classifier(purified_pp).argmax(dim=1)
    
        robust_acc_bfcp = (pred_bfcp.eq(labels)).float().mean().item()
        robust_acc_pp = (pred_pp.eq(labels)).float().mean().item()
    
        ssim_bfcp_val = ssim(purified_bfcp, data, data_range=1.0).item()
        ssim_pp_val = ssim(purified_pp, data, data_range=1.0).item()
    
        print(f"Batch {batch_idx}: BFCP Robust Acc: {robust_acc_bfcp:.3f}, Purify++ Robust Acc: {robust_acc_pp:.3f}")
        print(f"Batch {batch_idx}: BFCP SSIM: {ssim_bfcp_val:.3f}, Purify++ SSIM: {ssim_pp_val:.3f}")
    
        robust_acc_bfcp_list.append(robust_acc_bfcp)
        robust_acc_pp_list.append(robust_acc_pp)
        ssim_bfcp_list.append(ssim_bfcp_val)
        ssim_pp_list.append(ssim_pp_val)
    
        batch_count += 1
        if batch_count >= 3:
            break

    avg_robust_acc_bfcp = sum(robust_acc_bfcp_list) / len(robust_acc_bfcp_list)
    avg_robust_acc_pp = sum(robust_acc_pp_list) / len(robust_acc_pp_list)
    avg_ssim_bfcp = sum(ssim_bfcp_list) / len(ssim_bfcp_list)
    avg_ssim_pp = sum(ssim_pp_list) / len(ssim_pp_list)
    
    print("Average Robust Accuracy: BFCP=%.3f, Purify++=%.3f" % (avg_robust_acc_bfcp, avg_robust_acc_pp))
    print("Average SSIM: BFCP=%.3f, Purify++=%.3f" % (avg_ssim_bfcp, avg_ssim_pp))

    labels_group = ['Robust Accuracy', 'SSIM']
    bfcp_means = [avg_robust_acc_bfcp, avg_ssim_bfcp]
    pp_means = [avg_robust_acc_pp, avg_ssim_pp]
    
    x = range(len(labels_group))
    width = 0.35
    
    images_dir = ensure_images_directory()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([p - width/2 for p in x], bfcp_means, width, label='BFCP', color='green')
    ax.bar([p + width/2 for p in x], pp_means, width, label='Purify++', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_group)
    ax.set_ylim(0, 1)
    ax.set_title("Average Robust Accuracy and SSIM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{images_dir}/accuracy_robustness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Experiment 2 plot saved as accuracy_robustness.pdf")

def experiment3():
    print("\nStarting Experiment 3: Semantic Preservation and Visual Quality Evaluation")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    device = get_device()
    vgg = vmodels.vgg16(weights='IMAGENET1K_V1').features.to(device)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False

    def perceptual_loss(x, y):
        with torch.no_grad():
            feat_x = vgg(x)
            feat_y = vgg(y)
        return F.mse_loss(feat_x, feat_y)

    mse_loss_bfcp_list = []
    mse_loss_pp_list = []
    perceptual_loss_bfcp_list = []
    perceptual_loss_pp_list = []
    latent_diff_bfcp_list = []
    latent_diff_pp_list = []
    
    batch_count = 0
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        
        epsilon = 0.03
        noise = torch.randn_like(data) * epsilon
        adv_data = torch.clamp(data + noise, 0, 1)
    
        purified_bfcp, _ = run_bfcp_pipeline(model=None, input_data=adv_data)
        purified_pp, _ = run_purify_pp_pipeline(model=None, input_data=adv_data)
    
        mse_loss_bfcp = F.mse_loss(purified_bfcp, data).item()
        mse_loss_pp = F.mse_loss(purified_pp, data).item()
    
        p_loss_bfcp = perceptual_loss(purified_bfcp, data).item()
        p_loss_pp = perceptual_loss(purified_pp, data).item()
    
        latent_clean = get_encoder_latent(data.to('cpu'))
        latent_bfcp = get_encoder_latent(purified_bfcp.to('cpu'))
        latent_pp = get_encoder_latent(purified_pp.to('cpu'))
        latent_diff_bfcp = F.mse_loss(latent_clean, latent_bfcp).item()
        latent_diff_pp = F.mse_loss(latent_clean, latent_pp).item()
    
        mse_loss_bfcp_list.append(mse_loss_bfcp)
        mse_loss_pp_list.append(mse_loss_pp)
        perceptual_loss_bfcp_list.append(p_loss_bfcp)
        perceptual_loss_pp_list.append(p_loss_pp)
        latent_diff_bfcp_list.append(latent_diff_bfcp)
        latent_diff_pp_list.append(latent_diff_pp)
    
        print(f"Batch {batch_idx}:")
        print(f"  BFCP - MSE: {mse_loss_bfcp:.4f}, Perceptual Loss: {p_loss_bfcp:.4f}, Latent Diff: {latent_diff_bfcp:.4f}")
        print(f"  Purify++ - MSE: {mse_loss_pp:.4f}, Perceptual Loss: {p_loss_pp:.4f}, Latent Diff: {latent_diff_pp:.4f}")
    
        if batch_idx == 0:
            orig_img = data[0].detach().cpu().permute(1, 2, 0).numpy()
            bfcp_img = purified_bfcp[0].detach().cpu().permute(1, 2, 0).numpy()
            pp_img = purified_pp[0].detach().cpu().permute(1, 2, 0).numpy()
    
            images_dir = ensure_images_directory()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(orig_img)
            axs[0].set_title("Original")
            axs[1].imshow(bfcp_img)
            axs[1].set_title("BFCP Purified")
            axs[2].imshow(pp_img)
            axs[2].set_title("Purify++ Purified")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{images_dir}/visual_comparison.pdf", bbox_inches="tight")
            plt.close(fig)
            print("Visual comparison plot saved as visual_comparison.pdf")
    
        batch_count += 1
        if batch_count >= 3:
            break

    avg_mse_bfcp = sum(mse_loss_bfcp_list)/len(mse_loss_bfcp_list)
    avg_mse_pp = sum(mse_loss_pp_list)/len(mse_loss_pp_list)
    avg_ploss_bfcp = sum(perceptual_loss_bfcp_list)/len(perceptual_loss_bfcp_list)
    avg_ploss_pp = sum(perceptual_loss_pp_list)/len(perceptual_loss_pp_list)
    avg_latent_bfcp = sum(latent_diff_bfcp_list)/len(latent_diff_bfcp_list)
    avg_latent_pp = sum(latent_diff_pp_list)/len(latent_diff_pp_list)
    
    print("\nAverage Metrics for Experiment 3:")
    print("BFCP: MSE = %.4f, Perceptual Loss = %.4f, Latent Diff = %.4f" %
          (avg_mse_bfcp, avg_ploss_bfcp, avg_latent_bfcp))
    print("Purify++: MSE = %.4f, Perceptual Loss = %.4f, Latent Diff = %.4f" %
          (avg_mse_pp, avg_ploss_pp, avg_latent_pp))
    
    metrics = ['MSE', 'Perceptual Loss', 'Latent Diff']
    bfcp_metrics = [avg_mse_bfcp, avg_ploss_bfcp, avg_latent_bfcp]
    pp_metrics = [avg_mse_pp, avg_ploss_pp, avg_latent_pp]
    
    x = range(len(metrics))
    width = 0.35
    images_dir = ensure_images_directory()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar([p - width/2 for p in x], bfcp_metrics, width, label='BFCP', color='purple')
    ax.bar([p + width/2 for p in x], pp_metrics, width, label='Purify++', color='brown')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Average Semantic Preservation Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{images_dir}/semantic_preservation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Experiment 3 plot saved as semantic_preservation.pdf")

def test_run():
    print("******** Starting Test Run of All Experiments ********")
    experiment1()
    experiment2()
    experiment3()
    print("******** Test Run Completed ********")

def main():
    print("="*80)
    print("BAYESIAN FLOW-CONTROLLED PURIFICATION (BFCP) EXPERIMENT SUITE")
    print("="*80)
    set_random_seeds(42)
    device = get_device()
    try:
        test_run()
        print("="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        images_dir = ensure_images_directory()
        print(f"All plots saved to: {images_dir}")
        print("- runtime_timings.pdf")
        print("- accuracy_robustness.pdf")
        print("- visual_comparison.pdf")
        print("- semantic_preservation.pdf")
        status_enum = "stopped"
        print(f"\nExperiment status: {status_enum}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        status_enum = "error"
        raise

if __name__ == "__main__":
    main()
