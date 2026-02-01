import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from ppo import PPOAgent
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("Warning: PyTorch not found. Training will not work, but data collection might.")
    torch = None
    DEVICE = None

from drone_env import DroneEnv, Config, BlueEvasivePolicy

# --- Step 1: Collect Demonstrations ---

def plot_training_results(log_df, output_dir, mode):
    plt.figure(figsize=(12, 5))
    
    # Returns
    plt.subplot(1, 2, 1)
    plt.plot(log_df['epoch'], log_df['avg_return'], label='Avg Return')
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    plt.title(f'Training Return ({mode})')
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(log_df['epoch'], log_df['loss'], label='Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({mode})')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"training_plot_{mode}.png")
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")
    plt.close()

def collect_demonstrations(mode, num_episodes, output_dir):
    print(f"Collecting {num_episodes} Expert demos in {mode} mode...")
    env = DroneEnv(mode)
    expert = BlueEvasivePolicy()
    
    data = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            # Expert Action
            action = expert.get_action(obs, 'blue') # raw action (int or array)
            
            # Record State-Action
            state_vec = env.get_flat_state(obs) # The features our NN sees
            
            # Record
            record = {
                'episode_id': ep,
                'mode': mode,
                'state': state_vec,
                'action': action,
                'blue_pos': obs['blue'][0:2].copy(),
                'red_pos': obs['red'][0:2].copy()
            }
            data.append(record)
            
            # Step Env (assuming perfect execution for expert)
            obs, r, done, info = env.step(action)
    
    # Process for saving
    # We save a simpler dataframe for analysis, but return tensors for training
    print(f"Collected {len(data)} transitions.")
    
    # Save raw data
    states = np.stack([d['state'] for d in data])
    if mode == 'DISCRETE':
        actions = np.array([d['action'] for d in data])
    else:
        actions = np.stack([d['action'] for d in data])
        
    save_path = os.path.join(output_dir, f"expert_{mode}")
    if torch:
        torch.save({'states': states, 'actions': actions}, save_path + ".pt")
        print(f"Expert data saved to {save_path}.pt")
    else:
        np.savez(save_path + ".npz", states=states, actions=actions)
        print(f"Expert data saved to {save_path}.npz (NumPy format)")
    
    return states, actions

# --- Step 2: Behavior Cloning ---

def train_bc(mode, states, actions, output_dir, epochs=20):
    print(f"Training BC Policy for {mode}...")
    
    state_dim = states.shape[1]
    
    if mode == 'DISCRETE':
        action_dim = 9
        criterion = nn.CrossEntropyLoss()
        # Actions to LongTensor
        y_train = torch.tensor(actions, dtype=torch.long).to(DEVICE)
    else:
        action_dim = 2
        criterion = nn.MSELoss()
        y_train = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
        
    x_train = torch.tensor(states, dtype=torch.float32).to(DEVICE)
    
    # Use PPO Agent's internal model structure for compatibility
    agent = PPOAgent(state_dim, action_dim, mode)
    agent.model.to(DEVICE)
    
    optimizer = optim.Adam(agent.model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            
            # BC uses the Actor head directly
            if mode == 'DISCRETE':
                # model() returns dist, val. accessing actor directly is cleaner
                logits = agent.model.actor(agent.model.trunk(bx))
                loss = criterion(logits, by)
            else:
                # Continuous: predict mean action
                preds = torch.tanh(agent.model.actor_mean(agent.model.trunk(bx)))
                # Scale if needed? Our expert output is raw accel [-max, max]
                # But our env inputs are clipped.
                # If network outputs [-1,1], we should scale expert actions to [-1,1] or network to [max]
                # Simplest: The expert action stored is 'total_acc'. 
                # Let's assume network outputs [-1, 1], so we target expert_action / MAX_ACCEL
                target = by / Config.BLUE_MAX_ACCEL
                loss = criterion(preds, target)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
    # Save BC weights
    save_path = os.path.join(output_dir, f"bc_model_{mode}.pth")
    torch.save(agent.model.state_dict(), save_path)
    print(f"BC Model saved to {save_path}")
    
    return agent

# --- Step 3: RL Fine-tuning ---

def train_rl(mode, start_agent=None, timesteps=100000, output_dir="drone_data"):
    print(f"Fine-tuning with PPO ({mode})...")
    
    env = DroneEnv(mode)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    if start_agent:
        agent = start_agent
    else:
        # Load BC if exists, else scratch
        bc_path = os.path.join(output_dir, f"bc_model_{mode}.pth")
        agent = PPOAgent(state_dim, action_dim, mode)
        if os.path.exists(bc_path):
            print("Loading BC weights for initialization...")
            agent.model.load_state_dict(torch.load(bc_path))
            agent.model.to(DEVICE)
    
    # Training Loop
    steps_per_epoch = 2048 # Buffer size
    epochs = timesteps // steps_per_epoch
    
    ep_returns = []
    avg_dists = []
    
    obs = env.reset()
    curr_ep_ret = 0
    curr_ep_len = 0
    
    total_steps = 0
    
    training_logs = []
    
    for epoch in range(epochs):
        agent.clear_memory()
        
        # Collect Rollouts
        for t in range(steps_per_epoch):
            state_vec = env.get_flat_state(obs)
            s_tensor = torch.tensor(state_vec, dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                action, log_prob, val = agent.model.get_action(s_tensor)
            
            # Execute
            if mode == 'DISCRETE':
                act_env = action.item()
            else:
                # Network outputs [-1, 1], Scale to env limits
                act_env = action.cpu().numpy() * Config.BLUE_MAX_ACCEL
                
            obs, reward, done, info = env.step(act_env)
            
            # Store
            agent.store(state_vec, 
                        action.item() if mode=='DISCRETE' else action.cpu().numpy(),
                        reward, 
                        val.item(), 
                        log_prob.item(), 
                        done)
            
            curr_ep_ret += reward
            curr_ep_len += 1
            total_steps += 1
            
            if done:
                ep_returns.append(curr_ep_ret)
                avg_dists.append(info['distance'])
                obs = env.reset()
                curr_ep_ret = 0
                curr_ep_len = 0
                
        # Update
        # Value of last state for GAE
        last_state = env.get_flat_state(obs)
        ls_tensor = torch.tensor(last_state, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            _, last_val = agent.model(ls_tensor)
            
        loss = agent.update(last_val.item())
        
        # Log
        avg_ret = np.mean(ep_returns[-10:]) if ep_returns else 0.0
        print(f"RL Epoch {epoch+1}/{epochs} | Avg Return: {avg_ret:.2f} | Loss: {loss:.4f}")
        
        training_logs.append({
            'epoch': epoch + 1,
            'avg_return': avg_ret,
            'loss': loss
        })
        
    # Save RL weights
    save_path = os.path.join(output_dir, f"rl_model_{mode}.pth")
    torch.save(agent.model.state_dict(), save_path)
    print(f"RL Model saved to {save_path}")
    
    # Save Logs and Plot
    log_df = pd.DataFrame(training_logs)
    log_path = os.path.join(output_dir, f"training_log_{mode}.csv")
    log_df.to_csv(log_path, index=False)
    print(f"Training logs saved to {log_path}")
    
    plot_training_results(log_df, output_dir, mode)
    
    return agent, ep_returns

# --- Step 4: Full Evaluation ---

def evaluate_agents(mode, output_dir, visualize=False):
    print(f"\nEvaluating Agents ({mode} Mode)...")
    env = DroneEnv(mode)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    agents = {}
    
    # 1. Expert
    agents['Expert'] = BlueEvasivePolicy()
    
    # 2. BC
    bc_agent = PPOAgent(state_dim, action_dim, mode)
    bc_path = os.path.join(output_dir, f"bc_model_{mode}.pth")
    if os.path.exists(bc_path):
        bc_agent.model.load_state_dict(torch.load(bc_path))
        bc_agent.model.eval()
        agents['BC'] = bc_agent
    
    # 3. RL
    rl_agent = PPOAgent(state_dim, action_dim, mode)
    rl_path = os.path.join(output_dir, f"rl_model_{mode}.pth")
    if os.path.exists(rl_path):
        rl_agent.model.load_state_dict(torch.load(rl_path))
        rl_agent.model.eval()
        agents['BC+RL'] = rl_agent
        
    # Run Eval Loop
    results = []
    
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))
    
    for name, policy in agents.items():
        # Reset seed for fair comparison (same episodes for all agents)
        env.seed(Config.SEED)
        print(f"Testing {name}...")
        for i in range(50): # 50 Eval episodes per agent
            obs = env.reset()
            done = False
            trajectory = []
            
            while not done:
                if name == 'Expert':
                    action = policy.get_action(obs, 'blue')
                    act_env = action
                else:
                    # Neural Policy
                    state_vec = env.get_flat_state(obs)
                    s_tensor = torch.tensor(state_vec, dtype=torch.float32)
                    with torch.no_grad():
                        action, _, _ = policy.model.get_action(s_tensor, deterministic=True)
                    
                    if mode == 'DISCRETE':
                        act_env = action.item()
                    else:
                        act_env = action.numpy() * Config.BLUE_MAX_ACCEL
                
                obs_prev = obs
                obs, r, done, info = env.step(act_env)
                
                if visualize and name == 'BC+RL' and i < 5:
                    env.render(ax)
                    plt.pause(0.01)
                
                # Record specific items for dataset
                row = {
                    'run_type': name,
                    'episode_id': i,
                    't': env.t,
                    'step': env.step_count,
                    'mode': mode,
                    'blue_x': obs_prev['blue'][0], 'blue_y': obs_prev['blue'][1],
                    'red_x': obs_prev['red'][0], 'red_y': obs_prev['red'][1],
                    'distance': env.get_distance(),
                    'caught': 1 if info['outcome']=='caught' else 0
                }
                trajectory.append(row)
            
            results.extend(trajectory)
            
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"evaluation_results_{mode}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")
    
    # Print Stats
    params = df.groupby('run_type').agg({
        'caught': 'max', # Max of caught flag per episode is if it was caught
        'step': 'max',   # Length
        'distance': 'mean'
    })
    # Correct aggregation: group by run_type AND episode_id first
    episode_stats = df.groupby(['run_type', 'episode_id']).agg({
        'caught': 'max',
        'step': 'max',
        'distance': 'min' # Min distance achieved
    }).reset_index()
    
    summary = episode_stats.groupby('run_type').agg({
        'caught': 'mean', # Catch rate
        'step': 'mean',   # Avg survival time
        'distance': 'mean' # Avg min distance
    })
    
    print("\n--- Results Summary ---")
    print(summary)
    print("-----------------------")
    
    if visualize:
        plt.ioff()
        plt.close()

# --- Main CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='CONTINUOUS', choices=['CONTINUOUS', 'DISCRETE'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--collect_demos', action='store_true', help="Step 1: Generate Expert Data")
    parser.add_argument('--train_bc', action='store_true', help="Step 2: Train Behavior Cloning")
    parser.add_argument('--train_rl', action='store_true', help="Step 3: Fine-tune with RL")
    parser.add_argument('--eval', action='store_true', help="Step 4: Evaluate all agents")
    parser.add_argument('--all', action='store_true', help="Run all steps in order")
    parser.add_argument('--episodes_demo', type=int, default=2000)
    parser.add_argument('--steps_rl', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default="drone_data")
    parser.add_argument('--visualize', action='store_true', help="Visualize RL evaluation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    Config.SEED = args.seed
    
    if args.all:
        args.collect_demos = True
        args.train_bc = True
        args.train_rl = True
        args.eval = True
        
    if args.collect_demos:
        states, actions = collect_demonstrations(args.mode, args.episodes_demo, args.output_dir)
    elif args.train_bc: # If skipped demo collection but want to train, load files
        pt_path = os.path.join(args.output_dir, f"expert_{args.mode}.pt")
        if os.path.exists(pt_path):
            data = torch.load(pt_path)
            states, actions = data['states'], data['actions']
        else:
            print("Expert data not found! Run with --collect_demos first.")
            exit(1)
            
    bc_agent = None
    if args.train_bc:
        bc_agent = train_bc(args.mode, states, actions, args.output_dir)
        
    if args.train_rl:
        # Pass bc_agent to initialize RL agent, or it will load from disk
        train_rl(args.mode, start_agent=bc_agent, timesteps=args.steps_rl, output_dir=args.output_dir)
        
    if args.eval:
        evaluate_agents(args.mode, args.output_dir, args.visualize)
