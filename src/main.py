import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys

from preprocess import get_environment, generate_expert_data, get_device, set_random_seeds
from train import (
    PolicyNet, DiffusionModel, train_agent, train_desil_with_refinement,
    train_desil_without_refinement, profile_diffusion_inference
)
from evaluate import plot_and_save, save_bar_chart, ensure_images_directory

warnings.filterwarnings("ignore")

def experiment_performance_comparison(device):
    """Experiment 1: Performance Comparison in a Continuous Control Task"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Performance Comparison in a Continuous Control Task")
    print("="*80)
    
    env = get_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env.spec.id}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    
    bc_policy = PolicyNet(state_dim, action_dim).to(device)
    bc_optimizer = optim.Adam(bc_policy.parameters(), lr=7e-4)
    print("Training Baseline BC Agent (minimal iterations)...")
    bc_rewards = train_agent(env, bc_policy, bc_optimizer, use_diffusion=False, num_iterations=10)
    
    desil_policy = PolicyNet(state_dim, action_dim).to(device)
    desil_diffusion = DiffusionModel(state_dim, action_dim).to(device)
    desil_optimizer = optim.Adam(desil_policy.parameters(), lr=7e-4)
    print("Training DESIL Agent (minimal iterations)...")
    desil_rewards = train_agent(env, desil_policy, desil_optimizer, diffusion=desil_diffusion, use_diffusion=True, num_iterations=10)
    
    iterations = list(range(len(bc_rewards)))
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "reward_comparison.pdf")
    plot_and_save(iterations, [bc_rewards, desil_rewards], ["Baseline BC", "DESIL"],
                  "Iterations", "Episode Reward", "Reward Comparison", filename)
    
    print("Experiment 1 complete.\n")
    env.close()

def experiment_ablation_study(device):
    """Experiment 2: Ablation Study on the Self-Guided Refinement Loop"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Ablation Study on the Self-Guided Refinement Loop")
    print("="*80)
    
    env = get_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env.spec.id}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    
    policy_full = PolicyNet(state_dim, action_dim).to(device)
    diffusion_full = DiffusionModel(state_dim, action_dim).to(device)
    optimizer_full = optim.Adam(policy_full.parameters(), lr=7e-4)
    full_desil_rewards = train_desil_with_refinement(env, policy_full, optimizer_full, diffusion_full,
                                                     confidence_threshold=0.8, refine_phase_start=5, num_iterations=10)
    
    policy_no_refine = PolicyNet(state_dim, action_dim).to(device)
    diffusion_no_refine = DiffusionModel(state_dim, action_dim).to(device)
    optimizer_no_refine = optim.Adam(policy_no_refine.parameters(), lr=7e-4)
    no_refine_rewards = train_desil_without_refinement(env, policy_no_refine, optimizer_no_refine, diffusion_no_refine, num_iterations=10)
    
    iterations_ablation = list(range(len(full_desil_rewards)))
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "reward_ablation.pdf")
    plot_and_save(iterations_ablation, [full_desil_rewards, no_refine_rewards],
                  ["Full DESIL", "DESIL w/o Refinement"],
                  "Iterations", "Episode Reward", "Ablation: Reward Comparison", filename)
    
    print("Experiment 2 complete.\n")
    env.close()

def experiment_computational_efficiency(device):
    """Experiment 3: Evaluation of Computational Efficiency and Diffusion Loss Scaling"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Evaluation of Computational Efficiency and Diffusion Loss Scaling")
    print("="*80)
    
    env = get_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env.spec.id}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    
    policy = PolicyNet(state_dim, action_dim).to(device)
    diffusion = DiffusionModel(state_dim, action_dim).to(device)
    
    uniform_time = profile_diffusion_inference(env, policy, diffusion, use_selective=False, num_trials=20)
    selective_time = profile_diffusion_inference(env, policy, diffusion, confidence_threshold=0.8, use_selective=True, num_trials=20)
    
    modes = ["Uniform", "Selective"]
    times = [uniform_time, selective_time]
    
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "inference_latency.pdf")
    save_bar_chart(modes, times, "Inference Mode", "Average Inference Time (seconds)", 
                   "Diffusion Inference Latency Comparison", filename)
    
    print("Experiment 3 complete.\n")
    env.close()

def main():
    """Main experiment orchestrator"""
    print("="*80)
    print("DESIL EXPERIMENT SUITE")
    print("Diffusion-Enhanced Self-Guided Imitation Learning Implementation")
    print("="*80)
    
    set_random_seeds(42)
    device = get_device()
    
    try:
        print("\nStarting experimental evaluation...")
        
        experiment_performance_comparison(device)
        experiment_ablation_study(device)
        experiment_computational_efficiency(device)
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nExperiment Summary:")
        print("1. Performance Comparison: BC vs DESIL reward curves saved")
        print("2. Ablation Study: Self-guided refinement analysis completed")
        print("3. Computational Efficiency: Inference latency profiling evaluated")
        
        images_dir = ensure_images_directory()
        print(f"\nAll plots saved to: {images_dir}")
        print("- reward_comparison.pdf")
        print("- reward_ablation.pdf") 
        print("- inference_latency.pdf")
        
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
