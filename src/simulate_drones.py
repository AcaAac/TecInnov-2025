import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import time
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union

# --- Configuration ---

@dataclass
class Config:
    # Simulation
    DT: float = 0.05
    MAX_TIME: float = 30.0
    ARENA_SIZE: float = 3.0
    GRID_SIZE: int = 25  # For discrete mode
    CAPTURE_RADIUS: float = 0.05
    MIN_INIT_DIST: float = 0.3
    WALLS_MODE: bool = False  # True=hard walls, False=wrap-around (toroidal)

    # Physics - Red (Kamikaze: Fast, Agile, High Acceleration)
    RED_MASS: float = 1.0
    RED_MAX_ACCEL: float = 2.0
    RED_MAX_SPEED: float = 1.8
    RED_DRAG: float = 0.05

    # Physics - Blue (Fugitive: Heavy, Stable, Slow to turn)
    BLUE_MASS: float = 1.0
    BLUE_MAX_ACCEL: float = 1.0
    BLUE_MAX_SPEED: float = 0.8
    BLUE_DRAG: float = 0.2
    
    # Data
    NUM_EPISODES: int = 200
    OUTPUT_DIR: str = "drone_data"
    SEED: int = 42

    @property
    def MAX_STEPS(self) -> int:
        return int(self.MAX_TIME / self.DT)

CONFIG = Config()

# --- Helper Functions ---

def _get_toroidal_displacement(target_pos: np.ndarray,
                               my_pos: np.ndarray,
                               mode: str = 'CONTINUOUS') -> np.ndarray:
    """
    Calculate shortest displacement vector from my_pos to target_pos.

    In walls mode: Returns euclidean difference (target_pos - my_pos)
    In toroidal mode: Returns shortest displacement considering wrap-around

    Args:
        target_pos: Target position [x, y] (grid indices for discrete, coords for continuous)
        my_pos: Current position [x, y]
        mode: 'DISCRETE' or 'CONTINUOUS' (determines wrap boundary)

    Returns:
        Displacement vector pointing from my_pos toward target_pos via shortest path
    """
    diff = target_pos - my_pos

    # No correction needed in walls mode
    if CONFIG.WALLS_MODE:
        return diff

    # Determine wrap boundary based on mode
    boundary = CONFIG.GRID_SIZE if mode == 'DISCRETE' else CONFIG.ARENA_SIZE
    half_boundary = boundary / 2.0

    # Adjust each dimension for toroidal topology
    for i in range(2):
        if diff[i] > half_boundary:
            # Wrapping backward is shorter
            diff[i] -= boundary
        elif diff[i] < -half_boundary:
            # Wrapping forward is shorter
            diff[i] += boundary

    return diff

# --- Environment ---

class DroneEnv:
    def __init__(self, mode: str = 'CONTINUOUS'):
        self.mode = mode.upper()
        self.t = 0
        self.step_count = 0
        self.blue_state = None  # [x, y, vx, vy] or [row, col]
        self.red_state = None
        self.rng = np.random.RandomState(CONFIG.SEED)
        
        # Precompute grid cell size for discrete mode
        self.cell_size = CONFIG.ARENA_SIZE / CONFIG.GRID_SIZE

    def seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.t = 0
        self.step_count = 0
        
        # Initialize positions with minimum separation
        for _ in range(100):
            pos_blue = self.rng.rand(2) * CONFIG.ARENA_SIZE
            pos_red = self.rng.rand(2) * CONFIG.ARENA_SIZE

            # Check minimum separation distance
            if CONFIG.WALLS_MODE:
                dist = np.linalg.norm(pos_blue - pos_red)
            else:
                # Use toroidal distance for wrap-around mode
                diff = pos_blue - pos_red
                for i in range(2):
                    if abs(diff[i]) > CONFIG.ARENA_SIZE / 2:
                        diff[i] = CONFIG.ARENA_SIZE - abs(diff[i])
                dist = np.linalg.norm(diff)

            if dist >= CONFIG.MIN_INIT_DIST:
                break
        
        if self.mode == 'DISCRETE':
            # Quantize to grid indices
            self.blue_state = self._pos_to_idx(pos_blue)
            self.red_state = self._pos_to_idx(pos_red)
        else:
            # Continuous: [x, y, vx, vy]
            self.blue_state = np.array([pos_blue[0], pos_blue[1], 0.0, 0.0])
            self.red_state = np.array([pos_red[0], pos_red[1], 0.0, 0.0])
            
        return self._get_obs()

    def _pos_to_idx(self, pos):
        idx = (pos / self.cell_size).astype(int)
        return np.clip(idx, 0, CONFIG.GRID_SIZE - 1)

    def _idx_to_pos(self, idx):
        # Return center of cell
        return (idx + 0.5) * self.cell_size

    def step(self, action_blue, action_red):
        """
        Actions:
            Discrete: int 0-8 (movement directions)
            Continuous: np.array [ax, ay]
        """
        curr_dist = self.get_distance()
        caught = curr_dist <= CONFIG.CAPTURE_RADIUS

        if caught:
            return self._get_obs(), caught, True, {'outcome': 'caught'}
        
        if self.step_count >= CONFIG.MAX_STEPS:
            return self._get_obs(), caught, True, {'outcome': 'timeout'}

        # Update Dynamics
        if self.mode == 'DISCRETE':
            self.blue_state = self._step_discrete(self.blue_state, action_blue)
            self.red_state = self._step_discrete(self.red_state, action_red)
        else:
            self.blue_state = self._step_continuous(self.blue_state, action_blue, 'blue')
            self.red_state = self._step_continuous(self.red_state, action_red, 'red')

        self.t += CONFIG.DT
        self.step_count += 1
        
        # Check catch after move
        curr_dist = self.get_distance()
        caught = curr_dist <= CONFIG.CAPTURE_RADIUS
        done = caught or (self.step_count >= CONFIG.MAX_STEPS)
        
        return self._get_obs(), caught, done, {'outcome': 'caught' if caught else 'timeout'}

    def _step_discrete(self, state, action):
        # Actions: 0=stay, 1=up, 2=down, 3=left, 4=right, 5=ul, 6=ur, 7=dl, 8=dr
        moves = [
            (0,0), (0,1), (0,-1), (-1,0), (1,0), 
            (-1,1), (1,1), (-1,-1), (1,-1)
        ]
        dx, dy = moves[action]
        x, y = state
        if CONFIG.WALLS_MODE:
            # Hard walls: clamp to grid boundaries
            nx, ny = np.clip([x + dx, y + dy], 0, CONFIG.GRID_SIZE - 1)
        else:
            # Wrap-around: modulo arithmetic
            nx = (x + dx) % CONFIG.GRID_SIZE
            ny = (y + dy) % CONFIG.GRID_SIZE
        return np.array([nx, ny])

    def _step_continuous(self, state, action, agent_type):
        # state: x, y, vx, vy
        # action: fx, fy (force-like input)
        pos = state[0:2]
        vel = state[2:4]
        
        # Select constants based on agent type
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
        
        # Physics update
        vel += acc * CONFIG.DT
        vel *= (1.0 - drag * CONFIG.DT) # Damping
        
        speed = np.linalg.norm(vel)
        if speed > max_spd:
            vel = vel / speed * max_spd
            
        pos += vel * CONFIG.DT

        # Boundary handling
        if CONFIG.WALLS_MODE:
            # Hard walls with inelastic collision
            for i in range(2):
                if pos[i] < 0:
                    pos[i] = 0
                    vel[i] = 0  # Inelastic collision
                elif pos[i] > CONFIG.ARENA_SIZE:
                    pos[i] = CONFIG.ARENA_SIZE
                    vel[i] = 0
        else:
            # Wrap-around mode (toroidal topology)
            # Velocity and acceleration are preserved
            for i in range(2):
                pos[i] = pos[i] % CONFIG.ARENA_SIZE
                
        return np.array([pos[0], pos[1], vel[0], vel[1]])

    def get_distance(self):
        if self.mode == 'DISCRETE':
            p1 = self._idx_to_pos(self.blue_state)
            p2 = self._idx_to_pos(self.red_state)
        else:
            p1 = self.blue_state[0:2]
            p2 = self.red_state[0:2]

        if CONFIG.WALLS_MODE:
            # Euclidean distance
            return np.linalg.norm(p1 - p2)
        else:
            # Toroidal distance (shortest path through wrap-around)
            diff = p1 - p2
            for i in range(2):
                if abs(diff[i]) > CONFIG.ARENA_SIZE / 2:
                    # Shorter to go the other way
                    diff[i] = CONFIG.ARENA_SIZE - abs(diff[i])
            return np.linalg.norm(diff)

    def _get_obs(self):
        return {
            'blue': self.blue_state.copy(),
            'red': self.red_state.copy(),
            'mode': self.mode
        }

# --- Policies ---

class AgentPolicy:
    def get_action(self, obs, agent_type: str):
        raise NotImplementedError

class RedPursuitPolicy(AgentPolicy):
    def get_action(self, obs, agent_type='red'):
        mode = obs['mode']
        
        if mode == 'DISCRETE':
            # Greedy minimize distance
            # Try all 9 moves, pick best
            # This requires knowing the env dynamics ideally, 
            # or we simulate simple movement logic here.
            # We'll compute target direction vector in grid.
            my_pos = obs['red']
            target_pos = obs['blue']
            
            # Simple heuristic: move in direction of sign(diff)
            diff = _get_toroidal_displacement(target_pos, my_pos, mode='DISCRETE')
            # diff is [dx, dy] indices
            dx = int(np.sign(diff[0]))
            dy = int(np.sign(diff[1]))
            
            # Map dx,dy to action
            # moves mapping again:
            # (0,0)->0, (0,1)->1, (0,-1)->2, (-1,0)->3, (1,0)->4...
            # This mapping in env is:
            # 0:0,0; 1:0,1; 2:0,-1; 3:-1,0; 4:1,0..
            # Logic:
            if dx==0 and dy==0: return 0
            if dx==0 and dy==1: return 1
            if dx==0 and dy==-1: return 2
            if dx==-1 and dy==0: return 3
            if dx==1 and dy==0: return 4
            if dx==-1 and dy==1: return 5
            if dx==1 and dy==1: return 6
            if dx==-1 and dy==-1: return 7
            if dx==1 and dy==-1: return 8
            return 0 # Default

        else:
            # Continuous: PD controller
            my_state = obs['red']
            target_state = obs['blue']
            
            p_red = my_state[0:2]
            v_red = my_state[2:4]
            p_blue = target_state[0:2]
            v_blue = target_state[2:4]
            
            # Vector to target
            error_pos = _get_toroidal_displacement(p_blue, p_red, mode='CONTINUOUS')
            error_vel = v_blue - v_red
            
            # Proportional gain
            Kp = 5.0
            Kd = 1.0 # Damping
            
            desired_acc = Kp * error_pos + Kd * error_vel
            return desired_acc

class BlueEvasivePolicy(AgentPolicy):
    def __init__(self):
        self.rng = np.random.RandomState(CONFIG.SEED)
        
    def get_action(self, obs, agent_type='blue'):
        mode = obs['mode']
        
        if mode == 'DISCRETE':
            # Inverse of Red: move away
            my_pos = obs['blue']
            opp_pos = obs['red']
            
            diff = my_state = _get_toroidal_displacement(my_pos, opp_pos, mode='DISCRETE')
            # We want to increase distance.
            # Move in direction of sign(diff) to get further away
            dx = int(np.sign(diff[0]))
            dy = int(np.sign(diff[1]))

            # Wall avoidance heuristic for grid (only in walls mode)
            if CONFIG.WALLS_MODE:
                # If at detailed boundary, invert component
                if my_pos[0] == 0: dx = 1
                if my_pos[0] == CONFIG.GRID_SIZE-1: dx = -1
                if my_pos[1] == 0: dy = 1
                if my_pos[1] == CONFIG.GRID_SIZE-1: dy = -1

            # Convert to action
            if dx==0 and dy==0: 
                # If on top of each other (rare), pick random move
                return self.rng.randint(1, 9)
                
            # Mapping (same as Red but re-calculated)
            if dx==0 and dy==1: return 1
            if dx==0 and dy==-1: return 2
            if dx==-1 and dy==0: return 3
            if dx==1 and dy==0: return 4
            if dx==-1 and dy==1: return 5
            if dx==1 and dy==1: return 6
            if dx==-1 and dy==-1: return 7
            if dx==1 and dy==-1: return 8
            return 0

        else:
            # Continuous: Potential Fields
            my_state = obs['blue']
            opp_state = obs['red']
            
            p_blue = my_state[0:2]
            p_red = opp_state[0:2]
            
            # Repulse from Red
            diff = _get_toroidal_displacement(p_blue, p_red, mode='CONTINUOUS')
            dist = np.linalg.norm(diff) + 1e-6
            repulse_dir = diff / dist
            # Stronger repulsion when closer: 1/dist^2
            force_red = repulse_dir * (1 / (dist ** 2)) 
            
            # Repulse from Walls (only in walls mode)
            force_wall = np.zeros(2)
            if CONFIG.WALLS_MODE:
                margin = 0.2
                # Left wall (x=0)
                if p_blue[0] < margin: force_wall[0] += 1.0 / (p_blue[0] + 0.01)
                # Right wall (x=1)
                if p_blue[0] > 1.0 - margin: force_wall[0] -= 1.0 / (1.0 - p_blue[0] + 0.01)
                # Bottom wall (y=0)
                if p_blue[1] < margin: force_wall[1] += 1.0 / (p_blue[1] + 0.01)
                # Top wall (y=1)
                if p_blue[1] > 1.0 - margin: force_wall[1] -= 1.0 / (1.0 - p_blue[1] + 0.01)

            # Juke: Add perpendicular noise if red is close and fast
            juke_force = np.zeros(2)
            if dist < 0.2:
                # Perpendicular to red approach
                perp = np.array([-repulse_dir[1], repulse_dir[0]])
                if self.rng.rand() > 0.5: perp *= -1
                juke_force = perp * 2.0
            
            total_acc = force_red + force_wall * 0.05 + juke_force
            
            return total_acc

class BlueRLStub(AgentPolicy):
    def get_action(self, obs, agent_type='blue'):
        # Placeholder for RL model inference
        # For now, delegates to heuristic
        return BlueEvasivePolicy().get_action(obs)

# --- Simulation Logic ---

def run_episode(env: DroneEnv, 
                blue_policy: AgentPolicy, 
                red_policy: AgentPolicy, 
                episode_id: int,
                render_callback=None):
    
    obs = env.reset()
    done = False
    
    trajectory = []
    
    while not done:
        # Get actions
        act_blue = blue_policy.get_action(obs, 'blue')
        act_red = red_policy.get_action(obs, 'red')
        
        # Log before step (s_t, a_t) - here we log state s_t
        obs_curr = obs
        
        # Step
        obs, caught, done, info = env.step(act_blue, act_red)
        
        # Process logging
        # We need precise coordinates for dataset
        if env.mode == 'DISCRETE':
            b_pos = env._idx_to_pos(obs_curr['blue'])
            r_pos = env._idx_to_pos(obs_curr['red'])
            b_vel = [0,0] # No velocity in discrete state
            r_vel = [0,0]
        else:
            b_pos = obs_curr['blue'][0:2]
            r_pos = obs_curr['red'][0:2]
            b_vel = obs_curr['blue'][2:4]
            r_vel = obs_curr['red'][2:4]
            
        dist = np.linalg.norm(b_pos - r_pos)
        
        step_data = {
            'episode_id': episode_id,
            'step': env.step_count,
            'time': env.t,
            'mode': env.mode,
            'blue_x': b_pos[0], 'blue_y': b_pos[1],
            'red_x': r_pos[0], 'red_y': r_pos[1],
            'blue_vx': b_vel[0], 'blue_vy': b_vel[1],
            'red_vx': r_vel[0], 'red_vy': r_vel[1],
            'distance': dist,
            'caught': caught
        }
        trajectory.append(step_data)
        
        if render_callback:
            render_callback(b_pos, r_pos)
            
    return trajectory, caught

def run_batch_simulation(num_episodes, mode, show_anim=False):
    print(f"Starting {num_episodes} episodes in {mode} mode...")
    
    env = DroneEnv(mode=mode)
    blue_pol = BlueEvasivePolicy()
    red_pol = RedPursuitPolicy()
    
    all_data = []
    
    # Visualization Setup
    if show_anim:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(0, CONFIG.ARENA_SIZE)
        ax.set_ylim(0, CONFIG.ARENA_SIZE)
        ax.set_title(f"Drone Pursuit ({mode})")
        blue_dot, = ax.plot([], [], 'bo', markersize=10, label='Blue (Evader)')
        red_dot, = ax.plot([], [], 'ro', markersize=10, label='Red (Pursuer)')

        # Add capture radius circle around red drone
        capture_circle = plt.Circle((0, 0), CONFIG.CAPTURE_RADIUS,
                                   color='red', fill=False, linestyle='--',
                                   linewidth=2, alpha=0.5, label='Capture Radius')
        ax.add_patch(capture_circle)

        ax.legend()
        ax.set_aspect('equal')
        plt.ion()
        plt.show()

        def update_render(b, r):
            blue_dot.set_data([b[0]], [b[1]])
            red_dot.set_data([r[0]], [r[1]])
            # Update capture radius circle position to follow red drone
            capture_circle.center = (r[0], r[1])
            plt.pause(0.001)
    else:
        update_render = None

    start_time = time.time()
    
    stats_caught = 0
    stats_lens = []
    stats_min_dists = []

    for i in range(num_episodes):
        # Determine unique seed for this episode for reproducibility
        ep_seed = CONFIG.SEED + i
        env.seed(ep_seed)
        
        traj, caught = run_episode(env, blue_pol, red_pol, i, update_render)
        all_data.extend(traj)
        
        stats_caught += int(caught)
        stats_lens.append(len(traj))
        dists = [d['distance'] for d in traj]
        stats_min_dists.append(min(dists))
        
        if (i+1) % 50 == 0:
            print(f"Completed {i+1}/{num_episodes}...")
            
    if show_anim:
        plt.ioff()
        plt.close()
        
    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.2f}s")
    
    # Summary
    print("-" * 30)
    print(f"Summary ({mode}):")
    print(f"Catch Rate: {stats_caught/num_episodes*100:.1f}%")
    print(f"Avg Steps: {np.mean(stats_lens):.1f}")
    print(f"Avg Min Dist: {np.mean(stats_min_dists):.4f}")
    print("-" * 30)
    
    return pd.DataFrame(all_data), stats_lens, stats_min_dists

# --- Main Driver ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone-on-Drone Simulation")
    parser.add_argument('--mode', type=str, default='CONTINUOUS', choices=['CONTINUOUS', 'DISCRETE'], 
                        help='Simulation mode')
    parser.add_argument('--visualize', action='store_true', help='Show live animation (slow)')
    parser.add_argument('--episodes', type=int, default=CONFIG.NUM_EPISODES, help='Number of episodes')
    
    args = parser.parse_args()
    
    # Ensure output dir
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    # Run Simulation
    df, lens, min_dists = run_batch_simulation(args.episodes, args.mode, args.visualize)
    
    # Save Data
    csv_path = os.path.join(CONFIG.OUTPUT_DIR, "drone_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    try:
        import pyarrow
        pq_path = os.path.join(CONFIG.OUTPUT_DIR, "drone_dataset.parquet")
        df.to_parquet(pq_path, index=False)
        pq_msg = f"and {pq_path}"
    except ImportError:
        pq_msg = "(PyArrow not found, skipping Parquet)"

    print(f"Dataset saved to: {csv_path} {pq_msg}")
    
    # Post-run plotting (Histograms)
    # We only show this after run is done, blocking.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.hist(lens, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title("Episode Lengths (Steps)")
    ax1.set_xlabel("Steps")
    
    ax2.hist(min_dists, bins=20, color='salmon', edgecolor='black')
    ax2.set_title("Min Distances Achieved")
    ax2.set_xlabel("Distance")
    ax2.vlines(CONFIG.CAPTURE_RADIUS, 0, ax2.get_ylim()[1], colors='r', linestyles='dashed', label='Catch Radius')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(CONFIG.OUTPUT_DIR, "simulation_summary.png")
    plt.savefig(plot_path)
    print(f"Summary plot saved to: {plot_path}")
    
    # Show plot if running interactively
    # plt.show()
