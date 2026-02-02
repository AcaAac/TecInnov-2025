import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union

# --- Configuration ---

@dataclass
class Config:
    # Simulation
    DT: float = 0.05
    MAX_TIME: float = 30.0
    ARENA_SIZE: float = 1.0
    GRID_SIZE: int = 25  # For discrete mode
    CAPTURE_RADIUS: float = 0.05
    MIN_INIT_DIST: float = 0.3
    
    # Physics - Red (Kamikaze: Fast, Agile, High Acceleration)
    RED_MASS: float = 1.0
    RED_MAX_ACCEL: float = 3.0
    RED_MAX_SPEED: float = 1.8
    RED_DRAG: float = 0.05

    # Physics - Blue (Fugitive: Heavy, Stable, Slow to turn)
    BLUE_MASS: float = 2.0
    BLUE_MAX_ACCEL: float = 0.6
    BLUE_MAX_SPEED: float = 0.4
    BLUE_DRAG: float = 0.2
    
    # Data
    NUM_EPISODES: int = 200
    OUTPUT_DIR: str = "drone_data"
    SEED: int = 42

    @property
    def MAX_STEPS(self) -> int:
        return int(self.MAX_TIME / self.DT)

CONFIG = Config()

# --- Policies ---

class AgentPolicy:
    def get_action(self, obs, agent_type: str):
        raise NotImplementedError

class RedPursuitPolicy(AgentPolicy):
    def get_action(self, obs, agent_type='red'):
        mode = obs['mode']
        
        if mode == 'DISCRETE':
            # Greedy minimize distance
            my_pos = obs['red']
            target_pos = obs['blue']
            
            diff = target_pos - my_pos
            dx = int(np.sign(diff[0]))
            dy = int(np.sign(diff[1]))
            
            # Map dx,dy to action (0-8)
            # 0=stay, 1=up(0,1), 2=down(0,-1), 3=left(-1,0), 4=right(1,0)
            # 5=ul, 6=ur, 7=dl, 8=dr
            mapping = {
                (0,0):0, (0,1):1, (0,-1):2, (-1,0):3, (1,0):4,
                (-1,1):5, (1,1):6, (-1,-1):7, (1,-1):8
            }
            return mapping.get((dx, dy), 0)

        else:
            # Continuous: PD controller
            my_state = obs['red']
            target_state = obs['blue']
            
            p_red = my_state[0:2]
            v_red = my_state[2:4]
            p_blue = target_state[0:2]
            v_blue = target_state[2:4]
            
            # PD Control on relative motion
            error_pos = p_blue - p_red
            error_vel = v_blue - v_red
            Kp = 7.0
            Kd = 1.0
            desired_acc = Kp * error_pos + Kd * error_vel
            return desired_acc

class BlueEvasivePolicy(AgentPolicy):
    def __init__(self, seed=CONFIG.SEED):
        self.rng = np.random.RandomState(seed)
        
    def get_action(self, obs, agent_type='blue'):
        mode = obs['mode']
        
        if mode == 'DISCRETE':
            my_pos = obs['blue']
            opp_pos = obs['red']
            
            # Repel
            diff = my_pos - opp_pos
            if abs(diff[0]) < 1 and abs(diff[1]) < 1:
                # If very close, random panic move
                return self.rng.randint(1, 9)

            dx = int(np.sign(diff[0]))
            dy = int(np.sign(diff[1]))
            
            # Wall avoidance
            if my_pos[0] <= 1: dx = 1
            if my_pos[0] >= CONFIG.GRID_SIZE-2: dx = -1
            if my_pos[1] <= 1: dy = 1
            if my_pos[1] >= CONFIG.GRID_SIZE-2: dy = -1
            
            mapping = {
                (0,0):0, (0,1):1, (0,-1):2, (-1,0):3, (1,0):4,
                (-1,1):5, (1,1):6, (-1,-1):7, (1,-1):8
            }
            # Fallback if specific combo not found (e.g. wall override creates unique vec)
            # Just clamp to be safe
            dx = np.clip(dx, -1, 1)
            dy = np.clip(dy, -1, 1)
            return mapping.get((dx, dy), 0)

        else:
            # Continuous: Potential Fields
            my_state = obs['blue']
            opp_state = obs['red']
            
            p_blue = my_state[0:2]
            p_red = opp_state[0:2]
            
            # Repulse from Red
            diff = p_blue - p_red
            dist = np.linalg.norm(diff) + 1e-6
            repulse_dir = diff / dist
            force_red = repulse_dir * (0.5 / dist) 
            
            # Repulse from Walls
            force_wall = np.zeros(2)
            margin = 0.2
            if p_blue[0] < margin: force_wall[0] += 1.0 / (p_blue[0] + 0.1)
            if p_blue[0] > 1.0 - margin: force_wall[0] -= 1.0 / (1.0 - p_blue[0] + 0.1)
            if p_blue[1] < margin: force_wall[1] += 1.0 / (p_blue[1] + 0.1)
            if p_blue[1] > 1.0 - margin: force_wall[1] -= 1.0 / (1.0 - p_blue[1] + 0.1)
            
            # Juke
            juke_force = np.zeros(2)
            if dist < 0.25:
                # Perpendicular
                perp = np.array([-repulse_dir[1], repulse_dir[0]])
                if self.rng.rand() > 0.5: perp *= -1
                juke_force = perp * 2.5
            
            total_acc = force_red + force_wall * 0.1 + juke_force
            return total_acc

# --- Environment ---

class DroneEnv:
    def __init__(self, mode: str = 'CONTINUOUS'):
        self.mode = mode.upper()
        self.t = 0
        self.step_count = 0
        self.blue_state = None
        self.red_state = None
        self.rng = np.random.RandomState(CONFIG.SEED)
        self.cell_size = CONFIG.ARENA_SIZE / CONFIG.GRID_SIZE
        self.red_policy = RedPursuitPolicy()

    def seed(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self.red_policy = RedPursuitPolicy() # Stateless but good practice

    def reset(self):
        self.t = 0
        self.step_count = 0
        
        # Initialize positions
        for _ in range(100):
            pos_blue = self.rng.rand(2) * CONFIG.ARENA_SIZE
            pos_red = self.rng.rand(2) * CONFIG.ARENA_SIZE
            if np.linalg.norm(pos_blue - pos_red) >= CONFIG.MIN_INIT_DIST:
                break
        
        if self.mode == 'DISCRETE':
            self.blue_state = self._pos_to_idx(pos_blue)
            self.red_state = self._pos_to_idx(pos_red)
        else:
            self.blue_state = np.array([pos_blue[0], pos_blue[1], 0.0, 0.0])
            self.red_state = np.array([pos_red[0], pos_red[1], 0.0, 0.0])
            
        return self._get_obs()

    def _pos_to_idx(self, pos):
        idx = (pos / self.cell_size).astype(int)
        return np.clip(idx, 0, CONFIG.GRID_SIZE - 1)

    def _idx_to_pos(self, idx):
        return (idx + 0.5) * self.cell_size

    def step(self, action_blue):
        """
        Step function designed for RL training of Blue agent.
        Red agent moves automatically using its fixed policy.
        """
        obs_prev = self._get_obs()
        
        # 1. Get Red Action
        action_red = self.red_policy.get_action(obs_prev, 'red')
        
        # 2. Physics Step
        curr_dist = self.get_distance()
        caught = curr_dist <= CONFIG.CAPTURE_RADIUS

        if caught:
            # Already caught (shouldn't happen if check is done after step, but safe guard)
            return self._get_obs(), -10.0, True, {'outcome': 'caught'}
        
        if self.step_count >= CONFIG.MAX_STEPS:
            return self._get_obs(), 0.0, True, {'outcome': 'timeout'}

        if self.mode == 'DISCRETE':
            self.blue_state = self._step_discrete(self.blue_state, int(action_blue))
            self.red_state = self._step_discrete(self.red_state, int(action_red))
        else:
            self.blue_state = self._step_continuous(self.blue_state, action_blue, 'blue')
            self.red_state = self._step_continuous(self.red_state, action_red, 'red')

        self.t += CONFIG.DT
        self.step_count += 1
        
        # 3. Check Outcome
        new_dist = self.get_distance()
        caught = new_dist <= CONFIG.CAPTURE_RADIUS
        done = caught or (self.step_count >= CONFIG.MAX_STEPS)
        
        # 4. Calculate Reward
        reward = 0.0
        
        # Shaping: Reward for distance
        # Option A: Dense distance reward
        reward += new_dist * 0.1
        
        # Option B: Improvement reward (delta)
        # reward += (new_dist - curr_dist) * 10
        
        # Penalty for dying
        if caught:
            reward -= 50.0 # Big penalty
        else:
            # Survival bonus (small)
            reward += 0.05
            
        info = {
            'outcome': 'caught' if caught else 'timeout' if done else 'running',
            'distance': new_dist
        }
        
        return self._get_obs(), reward, done, info

    def _step_discrete(self, state, action):
        moves = [
            (0,0), (0,1), (0,-1), (-1,0), (1,0), 
            (-1,1), (1,1), (-1,-1), (1,-1)
        ]
        dx, dy = moves[action]
        x, y = state
        nx, ny = np.clip([x + dx, y + dy], 0, CONFIG.GRID_SIZE - 1)
        return np.array([nx, ny])

    def _step_continuous(self, state, action, agent_type):
        pos = state[0:2]
        vel = state[2:4]
        
        if agent_type == 'red':
            max_acc = CONFIG.RED_MAX_ACCEL
            max_spd = CONFIG.RED_MAX_SPEED
            drag = CONFIG.RED_DRAG
            mass = CONFIG.RED_MASS
        else:
            max_acc = CONFIG.BLUE_MAX_ACCEL
            max_spd = CONFIG.BLUE_MAX_SPEED
            drag = CONFIG.BLUE_DRAG
            mass = CONFIG.BLUE_MASS

        force = np.clip(action, -max_acc, max_acc)
        acc = force / mass
        vel += acc * CONFIG.DT
        vel *= (1.0 - drag * CONFIG.DT)
        
        speed = np.linalg.norm(vel)
        if speed > max_spd:
            vel = vel / speed * max_spd
            
        pos += vel * CONFIG.DT
        
        for i in range(2):
            if pos[i] < 0:
                pos[i] = 0
                vel[i] = 0
            elif pos[i] > CONFIG.ARENA_SIZE:
                pos[i] = CONFIG.ARENA_SIZE
                vel[i] = 0
                
        return np.array([pos[0], pos[1], vel[0], vel[1]])

    def get_distance(self):
        if self.mode == 'DISCRETE':
            p1 = self._idx_to_pos(self.blue_state)
            p2 = self._idx_to_pos(self.red_state)
        else:
            p1 = self.blue_state[0:2]
            p2 = self.red_state[0:2]
        return np.linalg.norm(p1 - p2)

    def _get_obs(self):
        return {
            'blue': self.blue_state.copy(),
            'red': self.red_state.copy(),
            'mode': self.mode
        }

    def get_flat_state(self, obs=None):
        """Returns a normalized flat feature vector for the network."""
        if obs is None: obs = self._get_obs()
        
        if self.mode == 'DISCRETE':
            # Normalize grid indices [0, N] -> [-1, 1]
            b = 2.0 * (obs['blue'] / CONFIG.GRID_SIZE) - 1.0
            r = 2.0 * (obs['red'] / CONFIG.GRID_SIZE) - 1.0
            return np.concatenate([b, r], dtype=np.float32)
        else:
            # Continuous: Pos [0,1]->[-1,1], Vel -> [-1,1] by dividing max speed
            b_pos = 2.0 * (obs['blue'][0:2] / CONFIG.ARENA_SIZE) - 1.0
            b_vel = obs['blue'][2:4] / CONFIG.BLUE_MAX_SPEED
            
            r_pos = 2.0 * (obs['red'][0:2] / CONFIG.ARENA_SIZE) - 1.0
            r_vel = obs['red'][2:4] / CONFIG.RED_MAX_SPEED
            
            return np.concatenate([b_pos, b_vel, r_pos, r_vel], dtype=np.float32)

    def get_state_dim(self):
        return 4 if self.mode == 'DISCRETE' else 8

    def get_action_dim(self):
        return 9 if self.mode == 'DISCRETE' else 2

    def render(self, ax=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if ax is None:
            # If no axis provided, create one (and show immediately if blocking)
            # But usually we want the caller to manage the figure for animation
            return

        ax.clear()
        ax.set_xlim(0, CONFIG.ARENA_SIZE)
        ax.set_ylim(0, CONFIG.ARENA_SIZE)
        ax.set_aspect('equal')
        
        # Get positions
        if self.mode == 'DISCRETE':
            p_blue = self._idx_to_pos(self.blue_state)
            p_red = self._idx_to_pos(self.red_state)
        else:
            p_blue = self.blue_state[0:2]
            p_red = self.red_state[0:2]
            
        # Draw Blue
        blue_circle = patches.Circle(p_blue, radius=0.02, color='blue', label='Blue')
        ax.add_patch(blue_circle)
        
        # Draw Red
        red_circle = patches.Circle(p_red, radius=0.02, color='red', label='Red')
        ax.add_patch(red_circle)
        
        # Draw Capture Radius around Red? Or just visualizing proximity
        capture_zone = patches.Circle(p_red, radius=CONFIG.CAPTURE_RADIUS, color='red', alpha=0.1)
        ax.add_patch(capture_zone)
        
        ax.set_title(f"Time: {self.t:.2f}s | Dist: {self.get_distance():.2f}")
        ax.legend(loc='upper right')
