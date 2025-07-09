import gymnasium as gym
import torch
import numpy as np
import random

def get_environment():
    """
    Get a continuous control environment for DESIL experiments.
    Falls back to simpler environment if MuJoCo is not available.
    """
    try:
        env = gym.make("HalfCheetah-v2")
        print("Using HalfCheetah-v2 environment")
    except (gym.error.DependencyNotInstalled, ImportError):
        env = gym.make("Pendulum-v1")
        print("Using Pendulum-v1 environment (MuJoCo not available)")
    return env

def generate_expert_data(env, num_samples=100):
    """
    Generate simulated expert demonstrations for the environment.
    In practice, this would be replaced with real expert data.
    """
    expert_data = []
    for _ in range(num_samples):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        action = np.zeros(env.action_space.shape)
        expert_data.append((state, action))
    return expert_data

def get_device():
    """Get the appropriate device (GPU if available, CPU otherwise)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU")
    return device

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
