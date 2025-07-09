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
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Device: {device}")
    print(f"Training iterations: 10 (minimal for testing)")
    
    print("\n" + "-"*60)
    print("TRAINING BASELINE BC AGENT")
    print("-"*60)
    bc_policy = PolicyNet(state_dim, action_dim).to(device)
    bc_optimizer = optim.Adam(bc_policy.parameters(), lr=7e-4)
    print(f"BC Policy architecture: {bc_policy}")
    print("Starting baseline behavioral cloning training...")
    bc_rewards = train_agent(env, bc_policy, bc_optimizer, use_diffusion=False, num_iterations=10)
    
    print("\n" + "-"*60)
    print("TRAINING DESIL AGENT")
    print("-"*60)
    desil_policy = PolicyNet(state_dim, action_dim).to(device)
    desil_diffusion = DiffusionModel(state_dim, action_dim).to(device)
    desil_optimizer = optim.Adam(desil_policy.parameters(), lr=7e-4)
    print(f"DESIL Policy architecture: {desil_policy}")
    print(f"DESIL Diffusion architecture: {desil_diffusion}")
    print("Starting DESIL training with diffusion confidence weighting...")
    desil_rewards = train_agent(env, desil_policy, desil_optimizer, diffusion=desil_diffusion, use_diffusion=True, num_iterations=10)
    
    print("\n" + "-"*60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("-"*60)
    bc_avg = np.mean(bc_rewards)
    desil_avg = np.mean(desil_rewards)
    bc_std = np.std(bc_rewards)
    desil_std = np.std(desil_rewards)
    
    print(f"Baseline BC - Average reward: {bc_avg:.2f} ± {bc_std:.2f}")
    print(f"DESIL Agent - Average reward: {desil_avg:.2f} ± {desil_std:.2f}")
    print(f"Performance improvement: {((desil_avg - bc_avg) / abs(bc_avg) * 100):.1f}%")
    
    iterations = list(range(len(bc_rewards)))
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "reward_comparison.pdf")
    plot_and_save(iterations, [bc_rewards, desil_rewards], ["Baseline BC", "DESIL"],
                  "Iterations", "Episode Reward", "Reward Comparison", filename)
    
    print(f"Performance comparison plot saved to: {filename}")
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
    print("Ablation study: Comparing DESIL with vs without self-guided refinement")
    
    print("\n" + "-"*60)
    print("TRAINING FULL DESIL (WITH SELF-GUIDED REFINEMENT)")
    print("-"*60)
    policy_full = PolicyNet(state_dim, action_dim).to(device)
    diffusion_full = DiffusionModel(state_dim, action_dim).to(device)
    optimizer_full = optim.Adam(policy_full.parameters(), lr=7e-4)
    print("Training DESIL with dual-phase approach (expert + self-guided refinement)...")
    full_desil_rewards = train_desil_with_refinement(env, policy_full, optimizer_full, diffusion_full,
                                                     confidence_threshold=0.8, refine_phase_start=5, num_iterations=10)
    
    print("\n" + "-"*60)
    print("TRAINING DESIL WITHOUT REFINEMENT (EXPERT-ONLY)")
    print("-"*60)
    policy_no_refine = PolicyNet(state_dim, action_dim).to(device)
    diffusion_no_refine = DiffusionModel(state_dim, action_dim).to(device)
    optimizer_no_refine = optim.Adam(policy_no_refine.parameters(), lr=7e-4)
    print("Training DESIL with expert demonstrations only (no self-guided refinement)...")
    no_refine_rewards = train_desil_without_refinement(env, policy_no_refine, optimizer_no_refine, diffusion_no_refine, num_iterations=10)
    
    print("\n" + "-"*60)
    print("ABLATION STUDY RESULTS")
    print("-"*60)
    full_avg = np.mean(full_desil_rewards)
    no_refine_avg = np.mean(no_refine_rewards)
    full_std = np.std(full_desil_rewards)
    no_refine_std = np.std(no_refine_rewards)
    
    print(f"Full DESIL (with refinement) - Average reward: {full_avg:.2f} ± {full_std:.2f}")
    print(f"DESIL w/o refinement - Average reward: {no_refine_avg:.2f} ± {no_refine_std:.2f}")
    print(f"Refinement contribution: {((full_avg - no_refine_avg) / abs(no_refine_avg) * 100):.1f}%")
    
    iterations_ablation = list(range(len(full_desil_rewards)))
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "reward_ablation.pdf")
    plot_and_save(iterations_ablation, [full_desil_rewards, no_refine_rewards],
                  ["Full DESIL", "DESIL w/o Refinement"],
                  "Iterations", "Episode Reward", "Ablation: Reward Comparison", filename)
    
    print(f"Ablation study plot saved to: {filename}")
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
    print("Evaluating computational efficiency of diffusion inference modes")
    
    policy = PolicyNet(state_dim, action_dim).to(device)
    diffusion = DiffusionModel(state_dim, action_dim).to(device)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print(f"Diffusion parameters: {sum(p.numel() for p in diffusion.parameters())}")
    
    print("\n" + "-"*60)
    print("PROFILING UNIFORM INFERENCE MODE")
    print("-"*60)
    uniform_time = profile_diffusion_inference(env, policy, diffusion, use_selective=False, num_trials=20)
    
    print("\n" + "-"*60)
    print("PROFILING SELECTIVE INFERENCE MODE")
    print("-"*60)
    selective_time = profile_diffusion_inference(env, policy, diffusion, confidence_threshold=0.8, use_selective=True, num_trials=20)
    
    print("\n" + "-"*60)
    print("COMPUTATIONAL EFFICIENCY RESULTS")
    print("-"*60)
    speedup = uniform_time / selective_time if selective_time > 0 else 1.0
    print(f"Uniform inference time: {uniform_time*1000:.3f} ms per sample")
    print(f"Selective inference time: {selective_time*1000:.3f} ms per sample")
    print(f"Speedup factor: {speedup:.2f}x")
    
    modes = ["Uniform", "Selective"]
    times = [uniform_time, selective_time]
    
    images_dir = ensure_images_directory()
    filename = os.path.join(images_dir, "inference_latency.pdf")
    save_bar_chart(modes, times, "Inference Mode", "Average Inference Time (seconds)", 
                   "Diffusion Inference Latency Comparison", filename)
    
    print(f"Computational efficiency plot saved to: {filename}")
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
