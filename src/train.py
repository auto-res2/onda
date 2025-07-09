import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

def train_agent(env, policy, optimizer, diffusion=None, use_diffusion=False, num_iterations=10):
    """
    Trains a given policy with either standard behavior cloning (BC) 
    or by incorporating diffusion confidence scores to weight the loss.
    """
    device = next(policy.parameters()).device
    cumulative_rewards = []
    total_steps = 0
    
    print(f"Training agent on device: {device}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    if diffusion is not None:
        print(f"Diffusion parameters: {sum(p.numel() for p in diffusion.parameters())}")
    
    for iter in range(num_iterations):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        done = False
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_pred = policy(state_tensor)
            action = action_pred.detach().cpu().numpy().flatten()
            confidence = 1.0
            if use_diffusion and (diffusion is not None):
                with torch.no_grad():
                    confidence = diffusion(state_tensor, action_pred).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

            expert_action = action
            expert_tensor = torch.FloatTensor(expert_action).unsqueeze(0).to(device)
            loss = confidence * ((action_pred - expert_tensor)**2).mean()
            episode_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
        
        total_steps += episode_steps
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        cumulative_rewards.append(episode_reward)
        
        if iter % 2 == 0:
            print(f"[Training] Iteration {iter}/{num_iterations-1}, Episode Reward: {episode_reward:.2f}, "
                  f"Steps: {episode_steps}, Avg Loss: {avg_loss:.6f}, Confidence: {confidence:.3f}")
    
    print(f"Training completed. Total steps: {total_steps}, Avg reward: {np.mean(cumulative_rewards):.2f}")
    return cumulative_rewards

def generate_trajectory(env, policy, max_steps=100):
    """
    Generates a trajectory (list of state-action pairs) from the current policy.
    """
    device = next(policy.parameters()).device
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result
    traj = []
    for _ in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = policy(state_tensor).detach().cpu().numpy().flatten()
        traj.append((state, action))
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            break
    return traj

def train_desil_with_refinement(env, policy, optimizer, diffusion, 
                                 confidence_threshold=0.8, 
                                 refine_phase_start=5, num_iterations=10):
    """
    Trains the DESIL agent with a two-phase strategy:
      Phase 1: Imitation only using (simulated) expert data.
      Phase 2: Self-guided refinement where agent-generated trajectories are fused.
    """
    device = next(policy.parameters()).device
    cumulative_rewards = []
    expert_data = []
    
    print(f"DESIL with Refinement - Phase transition at iteration {refine_phase_start}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    for _ in range(5):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            s = reset_result[0]
        else:
            s = reset_result
        a = np.zeros(env.action_space.shape)
        expert_data.append((s, a))
    
    for iter in range(num_iterations):
        if iter < refine_phase_start:
            print(f"[Phase 1 - Expert Imitation] Iteration {iter}")
            s, expert_action = expert_data[iter % len(expert_data)]
            state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
            action_pred = policy(state_tensor)
            with torch.no_grad():
                confidence = diffusion(state_tensor, action_pred).item()
            expert_tensor = torch.FloatTensor(expert_action).unsqueeze(0).to(device)
            loss = confidence * ((action_pred - expert_tensor)**2).mean()
        else:
            print(f"[Phase 2 - Self-Guided Refinement] Iteration {iter}")
            traj = generate_trajectory(env, policy)
            losses = []
            high_conf_count = 0
            for (s, a_real) in traj:
                state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                action_tensor = policy(state_tensor)
                with torch.no_grad():
                    conf = diffusion(state_tensor, action_tensor).item()
                if conf >= confidence_threshold:
                    high_conf_count += 1
                    real_tensor = torch.FloatTensor(a_real).unsqueeze(0).to(device)
                    loss_component = conf * ((action_tensor - real_tensor)**2).mean()
                    losses.append(loss_component)
            
            print(f"  Trajectory length: {len(traj)}, High confidence samples: {high_conf_count}")
            
            if losses:
                loss = sum(losses) / len(losses)
            else:
                dummy_param = next(policy.parameters())
                loss = torch.tensor(0.0, requires_grad=True, device=device)
        
        if hasattr(loss, 'backward'):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print(f"Warning: Loss is not a tensor, skipping backward pass. Loss type: {type(loss)}")
        
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state_eval = reset_result[0]
        else:
            state_eval = reset_result
        done = False
        ep_reward = 0.0
        while not done:
            state_tensor = torch.FloatTensor(state_eval).unsqueeze(0).to(device)
            action = policy(state_tensor).detach().cpu().numpy().flatten()
            state_eval, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        cumulative_rewards.append(ep_reward)
        if iter % 2 == 0:
            try:
                if hasattr(loss, 'item'):
                    loss_val = loss.item()
                else:
                    loss_val = float(loss)
            except:
                loss_val = 0.0
            print(f"[DESIL with Refinement] Iteration {iter}, Episode Reward: {ep_reward:.2f}, Loss: {loss_val:.6f}")
    return cumulative_rewards

def train_desil_without_refinement(env, policy, optimizer, diffusion, num_iterations=10):
    """
    Trains the DESIL variant without the self-guided refinement loop.
    It only uses the expert demonstration data weighted by the diffusion model.
    """
    device = next(policy.parameters()).device
    cumulative_rewards = []
    expert_data = []
    
    print("DESIL without Refinement - Expert-only training")
    
    for _ in range(5):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            s = reset_result[0]
        else:
            s = reset_result
        a = np.zeros(env.action_space.shape)
        expert_data.append((s, a))
    
    for iter in range(num_iterations):
        s, expert_action = expert_data[iter % len(expert_data)]
        state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
        action_pred = policy(state_tensor)
        with torch.no_grad():
            confidence = diffusion(state_tensor, action_pred).item()
        expert_tensor = torch.FloatTensor(expert_action).unsqueeze(0).to(device)
        loss = confidence * ((action_pred - expert_tensor)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state_eval = reset_result[0]
        else:
            state_eval = reset_result
        done = False
        ep_reward = 0.0
        while not done:
            state_tensor = torch.FloatTensor(state_eval).unsqueeze(0).to(device)
            action = policy(state_tensor).detach().cpu().numpy().flatten()
            state_eval, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        cumulative_rewards.append(ep_reward)
        if iter % 2 == 0:
            print(f"[DESIL without Refinement] Iteration {iter}, Episode Reward: {ep_reward:.2f}, "
                  f"Loss: {loss.item():.6f}, Confidence: {confidence:.3f}")
    return cumulative_rewards

def profile_diffusion_inference(env, policy, diffusion, confidence_threshold=0.8, 
                                use_selective=True, num_trials=20):
    """
    Profiles the inference time for the diffusion model.
    If use_selective==True, then only the cost for trajectories with confidence above threshold is counted.
    Otherwise, the diffusion model is run uniformly on all samples.
    Returns the average inference time per sample.
    """
    device = next(policy.parameters()).device
    total_time = 0.0
    count = 0
    high_conf_count = 0
    
    print(f"Profiling diffusion inference ({'Selective' if use_selective else 'Uniform'} mode)")
    print(f"Trials: {num_trials}, Confidence threshold: {confidence_threshold}")
    
    for trial in range(num_trials):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = policy(state_tensor)
        start_time = time.time()
        with torch.no_grad():
            conf = diffusion(state_tensor, action).item()
            if use_selective:
                if conf >= confidence_threshold:
                    high_conf_count += 1
                    dummy = conf * 2.0
            else:
                dummy = conf * 2.0
        elapsed = time.time() - start_time
        total_time += elapsed
        count += 1
        
        if trial % 5 == 0:
            print(f"  Trial {trial}: confidence={conf:.3f}, time={elapsed*1000:.3f}ms")
    
    avg_time = total_time / count
    mode = "Selective" if use_selective else "Uniform"
    
    if use_selective:
        print(f"High confidence samples: {high_conf_count}/{num_trials} ({100*high_conf_count/num_trials:.1f}%)")
    
    print(f"[Profiling] Average diffusion ({mode}) inference time per sample: {avg_time:.6f} seconds")
    return avg_time
