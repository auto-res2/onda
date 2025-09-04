import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

def plot_and_save(x_values, curves, labels, xlabel, ylabel, plot_title, filename):
    """
    Plots the given curves and saves the plot as a PDF with the given filename.
    """
    plt.figure(figsize=(8,6))
    for y in curves:
        plt.plot(x_values, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {filename}")

def save_bar_chart(categories, values, xlabel, ylabel, title, filename):
    """
    Saves a bar chart as PDF.
    """
    plt.figure(figsize=(6,4))
    sns.barplot(x=categories, y=values, palette="muted")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max(values) * 1.1)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved as {filename}")

def save_tsne_plot(features_combined, labels_vis, filename):
    """
    Creates and saves a t-SNE visualization plot as PDF.
    """
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_combined)
    
    plt.figure(figsize=(8,6))
    palette = {0:"blue", 1:"green", 2:"red"}
    for label in np.unique(labels_vis):
        idxs = np.where(labels_vis==label)
        plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], 
                    c=palette[label], label=["Clean", "Purify++", "ACDP"][label], alpha=0.7)
    plt.legend()
    plt.title("t-SNE of Feature Space for Different Image Types")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved as {filename}")

def save_adaptive_parameters_plot(lambda_log, noise_log, conf_log, filename):
    """
    Creates and saves a plot showing adaptive parameter evolution over diffusion steps.
    """
    plt.figure(figsize=(10,4))
    
    plt.subplot(1, 3, 1)
    plt.plot(lambda_log, marker='o')
    plt.title("Guidance Weight (λ) Over Steps")
    plt.xlabel("Diffusion Step")
    plt.ylabel("λ")
    
    plt.subplot(1, 3, 2)
    plt.plot(noise_log, marker='o', color='orange')
    plt.title("Noise Scaling Over Steps")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Noise Scaling")
    
    plt.subplot(1, 3, 3)
    plt.plot(conf_log, marker='o', color='green')
    plt.title("Classifier Confidence Over Steps")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Confidence")
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Adaptive parameters plot saved as {filename}")

def extract_features(feature_extractor, image_batch):
    """
    Extract features from images using a pretrained feature extractor.
    """
    with torch.no_grad():
        feats = feature_extractor(image_batch)
        feats = feats.view(feats.size(0), -1)
    return feats.cpu().numpy()

def compute_cosine_similarity(features_clean, features_purified):
    """
    Compute average cosine similarity between clean and purified features.
    """
    similarities = []
    for f_clean, f_purified in zip(features_clean, features_purified):
        sim = cosine_similarity(f_clean.reshape(1, -1), f_purified.reshape(1, -1))[0][0]
        similarities.append(sim)
    return np.mean(similarities)

def ensure_images_directory():
    """Ensure the images directory exists"""
    images_dir = ".research/iteration1/images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

def plot_loss_curve(loss_history, title, filename):
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved as {filename}")

def visualize_token_grid(image_np, token_resolution, filename='inference_token_grid.pdf'):
    H, W, _ = image_np.shape
    th, tw = token_resolution
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    for i in range(1, tw):
        ax.axvline(x=i * (W / tw), color='red', linestyle='--')
    for j in range(1, th):
        ax.axhline(y=j * (H / th), color='red', linestyle='--')
    plt.title("Token Grid Overlay")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Token grid visualization saved as {filename}")
