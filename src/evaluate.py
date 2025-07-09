import matplotlib.pyplot as plt
import numpy as np
import os

def plot_and_save(x_values, curves, labels, xlabel, ylabel, plot_title, filename):
    """
    Plots the given curves and saves the plot as a PDF with the given filename.
    """
    plt.figure()
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
    plt.figure()
    plt.bar(categories, values, color=["skyblue", "salmon"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved as {filename}")

def ensure_images_directory():
    """Ensure the images directory exists"""
    images_dir = ".research/iteration1/images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir
