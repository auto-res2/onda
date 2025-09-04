import torch
import torch.nn.functional as F
import torchattacks
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys

from preprocess import get_environment, generate_expert_data, get_device, set_random_seeds
from train import (
    get_classifier, get_feature_extractor, purify_purifypp, purify_acdp,
    adaptive_purification_with_logging
)
from evaluate import (
    save_bar_chart, save_tsne_plot, save_adaptive_parameters_plot,
    extract_features, compute_cosine_similarity, ensure_images_directory
)

warnings.filterwarnings("ignore")

def experiment_adversarial_purification_benchmarking(device):
    """Experiment 1: Adversarial Purification Benchmarking"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Adversarial Purification Benchmarking")
    print("="*80)
    
    test_loader = get_environment()
    classifier = get_classifier().to(device)
    
    print(f"Dataset: CIFAR-10 test set")
    print(f"Classifier: ResNet-18")
    print(f"Device: {device}")
    print(f"Attack: FGSM with eps=0.05")
    
    attack = torchattacks.FGSM(classifier, eps=0.05)
    
    batch_idx, (images, labels) = next(enumerate(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    print(f"Batch size: {images.size(0)}")
    print(f"Image shape: {images.shape}")
    
    print("\n" + "-"*60)
    print("GENERATING ADVERSARIAL EXAMPLES")
    print("-"*60)
    images_adv = attack(images, labels)
    print("Adversarial examples generated using FGSM")
    
    print("\n" + "-"*60)
    print("PURIFYING WITH PURIFY++")
    print("-"*60)
    purified_pp = purify_purifypp(images_adv)
    print("Purification completed using Purify++ method")
    
    print("\n" + "-"*60)
    print("PURIFYING WITH ACDP")
    print("-"*60)
    purified_acdp = purify_acdp(images_adv)
    print("Purification completed using ACDP method")
    
    print("\n" + "-"*60)
    print("EVALUATING CLASSIFICATION ACCURACY")
    print("-"*60)
    with torch.no_grad():
        preds_pp = classifier(purified_pp).argmax(dim=1)
        preds_acdp = classifier(purified_acdp).argmax(dim=1)
        
    acc_pp = (preds_pp == labels).float().mean().item()
    acc_acdp = (preds_acdp == labels).float().mean().item()
    
    print(f"Purify++ Accuracy: {acc_pp:.3f}")
    print(f"ACDP Accuracy: {acc_acdp:.3f}")
    if acc_pp > 0:
        print(f"Improvement: {((acc_acdp - acc_pp) / acc_pp * 100):.1f}%")
    else:
        print(f"Improvement: N/A (baseline accuracy is zero)")
    
    methods = ['Purify++', 'ACDP']
    accuracies = [acc_pp, acc_acdp]
    
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "adversarial_accuracy.pdf")
    save_bar_chart(methods, accuracies, "Purification Method", "Classification Accuracy",
                   "Adversarial Purification Accuracy Comparison", filename)
    
    print(f"Accuracy comparison plot saved to: {filename}")
    print("Experiment 1 complete.\n")

def experiment_feature_preservation_evaluation(device):
    """Experiment 2: Evaluating Feature Preservation"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Evaluating Feature Preservation")
    print("="*80)
    
    test_loader = get_environment()
    classifier = get_classifier().to(device)
    feature_extractor = get_feature_extractor().to(device)
    
    print(f"Dataset: CIFAR-10 test set")
    print(f"Feature extractor: Pretrained ResNet-18")
    print(f"Device: {device}")
    
    attack = torchattacks.FGSM(classifier, eps=0.05)
    
    images, _ = next(iter(test_loader))
    images = images.to(device)
    
    print(f"Batch size: {images.size(0)}")
    
    print("\n" + "-"*60)
    print("GENERATING ADVERSARIAL EXAMPLES")
    print("-"*60)
    images_adv = attack(images, torch.zeros(images.size(0), dtype=torch.long, device=device))
    print("Adversarial examples generated")
    
    print("\n" + "-"*60)
    print("PURIFYING IMAGES")
    print("-"*60)
    purified_pp = purify_purifypp(images_adv)
    purified_acdp = purify_acdp(images_adv)
    print("Images purified using both methods")
    
    print("\n" + "-"*60)
    print("EXTRACTING FEATURES")
    print("-"*60)
    features_clean = extract_features(feature_extractor, images)
    features_pp = extract_features(feature_extractor, purified_pp)
    features_acdp = extract_features(feature_extractor, purified_acdp)
    print(f"Features extracted - Shape: {features_clean.shape}")
    
    print("\n" + "-"*60)
    print("COMPUTING COSINE SIMILARITIES")
    print("-"*60)
    sim_pp = compute_cosine_similarity(features_clean, features_pp)
    sim_acdp = compute_cosine_similarity(features_clean, features_acdp)
    
    print(f"Purify++ Average Feature Cosine Similarity: {sim_pp:.4f}")
    print(f"ACDP Average Feature Cosine Similarity: {sim_acdp:.4f}")
    if sim_pp > 0:
        print(f"ACDP improvement: {((sim_acdp - sim_pp) / sim_pp * 100):.1f}%")
    else:
        print(f"ACDP improvement: N/A (baseline similarity is zero)")
    
    print("\n" + "-"*60)
    print("GENERATING T-SNE VISUALIZATION")
    print("-"*60)
    features_combined = np.concatenate([features_clean, features_pp, features_acdp], axis=0)
    labels_vis = np.concatenate([np.zeros(len(features_clean)), 
                                 np.ones(len(features_pp)), 
                                 2*np.ones(len(features_acdp))]).astype(int)
    
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "tsne_feature_preservation.pdf")
    save_tsne_plot(features_combined, labels_vis, filename)
    
    print("Experiment 2 complete.\n")

def experiment_adaptive_guidance_analysis(device):
    """Experiment 3: Impact Analysis of Adaptive Guidance and Randomness Control"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Impact Analysis of Adaptive Guidance and Randomness Control")
    print("="*80)
    
    test_loader = get_environment()
    classifier = get_classifier().to(device)
    
    print(f"Dataset: CIFAR-10 test set")
    print(f"Classifier: ResNet-18")
    print(f"Device: {device}")
    print(f"Adaptive purification steps: 50")
    
    attack = torchattacks.FGSM(classifier, eps=0.05)
    
    images_batch, _ = next(iter(test_loader))
    images_batch = images_batch.to(device)
    
    print(f"Batch size: {images_batch.size(0)}")
    
    print("\n" + "-"*60)
    print("GENERATING ADVERSARIAL EXAMPLES")
    print("-"*60)
    images_adv = attack(images_batch, torch.zeros(images_batch.size(0), dtype=torch.long, device=device))
    print("Adversarial examples generated")
    
    print("\n" + "-"*60)
    print("RUNNING ADAPTIVE PURIFICATION WITH LOGGING")
    print("-"*60)
    purified_adaptive, lambda_log, noise_log, conf_log = adaptive_purification_with_logging(
        images_adv, classifier, num_steps=50)
    
    print("Adaptive purification completed with parameter logging")
    print(f"Initial confidence: {conf_log[0]:.3f}")
    print(f"Final confidence: {conf_log[-1]:.3f}")
    print(f"Average guidance weight: {np.mean(lambda_log):.3f}")
    print(f"Average noise scaling: {np.mean(noise_log):.3f}")
    
    print("\n" + "-"*60)
    print("SAVING ADAPTIVE PARAMETER PLOTS")
    print("-"*60)
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "adaptive_parameters.pdf")
    save_adaptive_parameters_plot(lambda_log, noise_log, conf_log, filename)
    
    print("Experiment 3 complete.\n")

def main():
    """Main experiment orchestrator"""
    print("="*80)
    print("ACDP EXPERIMENT SUITE")
    print("Adaptive Classifier-Diffusion Purification Implementation")
    print("="*80)
    
    set_random_seeds(42)
    device = get_device()
    
    try:
        print("\nStarting experimental evaluation...")
        
        experiment_adversarial_purification_benchmarking(device)
        experiment_feature_preservation_evaluation(device)
        experiment_adaptive_guidance_analysis(device)
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nExperiment Summary:")
        print("1. Adversarial Purification: Purify++ vs ACDP accuracy comparison")
        print("2. Feature Preservation: Cosine similarity and t-SNE analysis")
        print("3. Adaptive Guidance: Parameter evolution during purification")
        
        images_dir = ensure_images_directory()
        print(f"\nAll plots saved to: {images_dir}")
        print("- adversarial_accuracy.pdf")
        print("- tsne_feature_preservation.pdf") 
        print("- adaptive_parameters.pdf")
        
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
