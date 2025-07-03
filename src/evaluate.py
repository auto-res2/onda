import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def save_training_loss_plot(loss_curve, filename):
    """Save joint training loss curve as PDF"""
    plt.figure()
    plt.plot(loss_curve, marker='o')
    plt.title("Joint Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Joint training loss curve saved to {filename}")

def save_adaptive_steps_plot(all_steps, filename):
    """Save adaptive diffusion steps distribution as PDF"""
    plt.figure()
    plt.hist(all_steps, bins=range(1, max(all_steps)+2), edgecolor='black')
    plt.title("Distribution of Adaptive Diffusion Steps")
    plt.xlabel("Steps Taken")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Adaptive diffusion steps histogram saved to {filename}")

def save_confidence_progression_plot(conf_history, filename):
    """Save confidence progression plot as PDF"""
    plt.figure()
    plt.plot(conf_history, marker='o')
    plt.title("Classifier Confidence Progression (First Image)")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Max Softmax Confidence")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Confidence progression plot saved to {filename}")

def save_robustness_tradeoff_plot(epsilons, accuracies, filename):
    """Save robustness trade-off plot as PDF"""
    plt.figure()
    plt.plot(epsilons, accuracies, marker='o')
    plt.title("Inference Latency and Robustness Trade-off")
    plt.xlabel("Epsilon (Attack Strength)")
    plt.ylabel("Adversarial Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Robustness trade-off plot saved to {filename}")

def ensure_images_directory():
    """Ensure the images directory exists"""
    images_dir = ".research/iteration1/images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir
