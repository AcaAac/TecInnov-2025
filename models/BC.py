import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class BehaviourClonning(nn.Module):
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
        self.head = nn.Linear(128, 1)

        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):

        x = self.trunk(state)

        value = self.head(x)
        

        mean = torch.tanh(self.actor_mean(x)) # Bound output [-1, 1], careful with scaling later if needed
        # For continuous, we want output to represent acceleration [-max, max].
        # We'll stick to raw network output ~[-1, 1] and scale in env wrapper or here.
        # Using tanh gives nicer bounded actions for movement.
        std = torch.exp(self.actor_logstd.clamp(-2, 1))
        dist = Normal(mean, std)
            
        return dist, value

    def get_action(self, state, deterministic=False):
        dist, value = self(state)
        
        if deterministic:
            action = dist.loc
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)

            
        return action, log_prob, value
