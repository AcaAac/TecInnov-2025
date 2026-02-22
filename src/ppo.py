import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np

# --- Neural Networks ---

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, mode='CONTINUOUS'):
        super().__init__()
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        
        # Critic Head
        self.critic = nn.Linear(128, 1)
        
        # Actor Head
        if mode == 'DISCRETE':
            self.actor = nn.Linear(128, action_dim)
        else:
            self.actor_mean = nn.Linear(128, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.trunk(state)
        value = self.critic(x)
        
        if self.mode == 'DISCRETE':
            logits = self.actor(x)
            dist = Categorical(logits=logits)
        else:
            mean = torch.tanh(self.actor_mean(x)) # Bound output [-1, 1], careful with scaling later if needed
            # For continuous, we want output to represent acceleration [-max, max].
            # We'll stick to raw network output ~[-1, 1] and scale in env wrapper or here.
            # Using tanh gives nicer bounded actions for movement.
            std = torch.exp(self.actor_logstd.clamp(-2, 1))
            dist = Normal(mean, std)
            
        return dist, value

    def get_action(self, state, deterministic=False):
        """Returns action, log_prob, value"""
        dist, value = self(state)
        
        if deterministic:
            if self.mode == 'DISCRETE':
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.loc
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        if self.mode != 'DISCRETE':
            log_prob = log_prob.sum(dim=-1)
            
        return action, log_prob, value

# --- PPO Agent ---

class PPOAgent:
    def __init__(self, state_dim, action_dim, mode='CONTINUOUS', lr=3e-4, gamma=0.99, clip_ratio=0.2, target_kl=0.01):
        self.mode = mode
        self.model = ActorCritic(state_dim, action_dim, mode)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = 0.95
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        
        # Buffer
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = [] # Raw rewards
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, last_val):
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_val])
        dones = np.array(self.dones)
        
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        
        advs = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advs[t] = gae
            
        returns = advs + values[:-1]
        return torch.tensor(advs, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, last_val, epochs=10, batch_size=64):
        # Determine device
        device = next(self.model.parameters()).device
        
        # Prepare data
        advs, returns = self.compute_gae(last_val)
        advs = advs.to(device)
        returns = returns.to(device)
        
        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
        
        # Actions handling
        if self.mode == 'DISCRETE':
            actions = torch.tensor(np.array(self.actions), dtype=torch.long).to(device)
        else:
            actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(device)
            
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32).to(device)
        
        dataset_size = len(states)
        metrics = {
            'loss_total': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_frac': 0.0
        }
        updates = 0
        early_stop = False

        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                s_batch = states[idx]
                a_batch = actions[idx]
                ret_batch = returns[idx]
                adv_batch = advs[idx]
                old_logp_batch = old_log_probs[idx]
                
                # Forward
                dist, v_pred = self.model(s_batch)
                
                # Value Loss
                v_loss = F.mse_loss(v_pred.squeeze(-1), ret_batch)
                
                # Policy Loss
                log_prob = dist.log_prob(a_batch)
                if self.mode != 'DISCRETE': log_prob = log_prob.sum(dim=-1)
                
                ratio = torch.exp(log_prob - old_logp_batch)
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_batch
                p_loss = -(torch.min(ratio * adv_batch, clip_adv)).mean()
                approx_kl = (old_logp_batch - log_prob).mean()
                clip_frac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                
                # Entropy Bonus
                entropy = dist.entropy().mean()
                
                loss = p_loss + 0.5 * v_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grads usually good practice
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                metrics['loss_total'] += loss.item()
                metrics['policy_loss'] += p_loss.item()
                metrics['value_loss'] += v_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += approx_kl.item()
                metrics['clip_frac'] += clip_frac.item()
                updates += 1

                if approx_kl.item() > 1.5 * self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        if updates > 0:
            for key in metrics:
                metrics[key] /= updates
        metrics['updates'] = updates
        metrics['early_stop'] = float(early_stop)

        self.clear_memory()
        return metrics
