
import random
import pygame
import numpy as np
from PIL import Image
from enum import Enum
from io import BytesIO
from typing import List
from os import makedirs
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

import gymnasium as gym

'''
References: 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

'''

class Directions(Enum):
	NORTH 		= (0, 1)
	NORTH_EAST 	= (1, 1)
	EAST 		= (1, 0)
	SOUTH_EAST 	= (1, -1)
	SOUTH 		= (0, -1)
	SOUTH_WEST 	= (-1, -1)
	WEST 		= (-1, 0)
	NORTH_WEST 	= (-1, 1)
	STAY 		= (0, 0)

class Actions(Enum):
	MOVE_NORTH 		= Directions.NORTH
	MOVE_NORTH_EAST = Directions.NORTH_EAST
	MOVE_EAST 		= Directions.EAST
	MOVE_SOUTH_EAST = Directions.SOUTH_EAST
	MOVE_SOUTH 		= Directions.SOUTH
	MOVE_SOUTH_WEST = Directions.SOUTH_WEST
	MOVE_WEST 		= Directions.WEST
	MOVE_NORTH_WEST = Directions.NORTH_WEST
	STAY 			= Directions.STAY
	
	def get_direction_xy(self):
		'''get the direction vector directly'''
		return self.value.value 
	
	@classmethod
	def get_by_index(cls, index):
		'''get the Enum name and value by by index: `Actions.get_by_index(0) = MOVE_NORTH`'''
		return list(cls)[index]    
	
	@classmethod
	def get_direction_by_index(cls, index):
		'''get the direction vector directly from index: `Actions.get_direction_by_index(0) = (0, 1)`'''
		return list(cls)[index].value.value
	
	@property
	def index(self):
		'''get the index: `Actions.MOVE_NORTH.index = 0`'''
		return list(Observations).index(self)
	
class Observations(Enum):
	NORTH 	= Directions.NORTH
	EAST 	= Directions.EAST
	SOUTH 	= Directions.SOUTH
	WEST 	= Directions.WEST

	def get_direction_xy(self):
		'''get the direction vector directly'''
		return self.value.value 
	
	@classmethod
	def get_by_index(cls, index):
		'''get the Enum name and value by by index: `Observations.get_by_index(0) = NORTH`'''
		return list(cls)[index]    
	
	@classmethod
	def get_direction_by_index(cls, index):
		'''
		get the direction vector directly from index: `Observations.get_direction_by_index(0) = (0, 1)`
		we use this to map the environment observation outputs array back to Observation(Enum)
		'''
		return list(cls)[index].value
	
	@property
	def index(self):
		'''get the index: `Observations.NORTH.index = 0`'''
		return list(Observations).index(self)

class GridWorldEnv(gym.Env):
	metadata = {'render_modes':['rgb_array']}
	def __init__(self, N: int = 5, obstacle_ratio=0.3, render_mode='rgb_array'):
		'''
		Gridworld is assumed to be NxN, entities modeled are the robot (agent) and walls (obstacles). 
		This class encapsulates:
		- gridworld generation, and corresponding transition and observation models/matrices
		- gridworld state transitions and agent observations
		- gridworld and observations rendering 
		'''
		
		## Grid properties
		# the size of the NxN grid
		self.N = N
		self.obstacle_ratio = obstacle_ratio 

		# observation space: dict of agent's position and sensor observations
		self.observation_space = gym.spaces.MultiBinary(4) # binary array corresponding to NESW obstacle detection
		self.agent_pos_space = gym.spaces.Box(0, N - 1, shape=(2,), dtype=int), # agent position cell (x,y)

		# initial agent location and senseor observation
		# randomly chosen in reset() and updated in step()
		self.agent_pos = np.array([-1, -1], dtype=np.int32)
		self.sensor_obs = np.zeros(shape=self.observation_space.shape[0], dtype=np.int8)
		self.all_obstacle_pos = np.zeros(shape=[self.N-1,self.N-1], dtype=np.int32)
		self.all_free_cell_pos = None 

		# 8+1 actions, move to any of the 8 neighoring cells or stay
		self.action_space = gym.spaces.Discrete(8+1)
		self.valid_actions =  {member.name for member in Actions}
		self.prev_action = Actions.STAY

		## environment models
		self.grid = np.zeros((self.N,self.N))
		
		# get the grid indices in array with shape (M,2) , where M is the number of all cells (i.e NxN)
		self.grid_indices = np.indices(self.grid.shape).reshape(2, -1).T

		self.num_states = self.N * self.N
		self.transition_matrix = self.init_transition_matrix()
		self.transition_matrix_action = None # optional, no action-conditioned model by default

		obs_vector_size = self.observation_space.shape[0]
		self.num_observations = 2**(obs_vector_size)
		# generate all possible combinations of binary observations
		#  i.e [(0,0,0,0),(0,0,0,1)...] all the way to 2^4
		# NOTE: this is not really necessary (we could just compute for the specific world), but kept for completeness
		self.all_possible_observations = list(product([0, 1], repeat=obs_vector_size))
		self.observation_matrix = self.init_observation_matrix()
		
		# reset env properties
		self.reset_env_model_properties()

		## rendering settings
		self.cur_step = 1
		self.window_size = 512  # size of rendering window
		self.render_mode = render_mode
	
	def reset_env_model_properties(self):
		'''reset all caches of the environment'''
		self._cached_transition_matrix = self.init_transition_matrix()
		self._cached_observation_matrix = self.init_observation_matrix()
		self._cached_transition_matrix_lookup = None
		self._cached_transition_matrix_reverse_lookup = None
		self._cached_observation_id_lookup = None
		self._cached_observation_id_reverse_lookup = None 
		self.transition_matrix_action = None # optional, no action-conditioned model by default

	@property
	def transition_matrix_lookup(self):
		'''lookup of (x,y) cell given cell_id'''
		if self._cached_transition_matrix_lookup is None:
			self._cached_transition_matrix_lookup = {cell_id: xy for cell_id, xy in enumerate(self.grid_indices) }
		return self._cached_transition_matrix_lookup
	
	@property
	def transition_matrix_reverse_lookup(self):
		'''lookup of cell_id given (x,y) -- NOTE: numpy arrays are not hashable so we need to convert them to tuple'''
		if self._cached_transition_matrix_reverse_lookup is None:
			self._cached_transition_matrix_reverse_lookup = {tuple(xy.tolist()): cell_id for cell_id, xy in self.transition_matrix_lookup.items()}
		return self._cached_transition_matrix_reverse_lookup
	
	@property
	def observation_id_lookup(self):
		'''lookup observation-id given raw-observation'''
		if self._cached_observation_id_lookup is None:
			self._cached_observation_id_lookup = {obs_raw:obs_id for obs_id, obs_raw in enumerate(self.all_possible_observations)}
		return self._cached_observation_id_lookup
	
	@property
	def observation_id_reverse_lookup(self):
		'''lookup raw-observation given observation-id'''
		if self._cached_observation_id_reverse_lookup is None:
			self._cached_observation_id_reverse_lookup = {obs_id:obs_raw for obs_raw, obs_id in self.observation_id_lookup.items()}
		return self._cached_observation_id_reverse_lookup

	def init_transition_matrix(self):
		'''initialize with zeros'''
		return np.zeros((self.num_states, self.num_states))
	
	def init_observation_matrix(self):
		'''initialize with zeros'''
		return np.zeros((self.num_states, self.num_observations))
	
	def _get_state_id_from_pos(self, pos: np.ndarray) -> int:
		''' map (x,y) position to a state id '''
		pos_tuple = tuple(pos.tolist())
		return self.transition_matrix_reverse_lookup[pos_tuple]
	
	def get_priviledged_info(self):
		'''priviledged environment/simulator data'''
		# agent_position is considered a ground-truth and should not be used for self-localization
		return {'agent_position': self.agent_pos}
	
	def get_model_info(self):
		'''HMM model info'''
		return {
			'transition_matrix': self.transition_matrix,
			'observation_matrix': self.observation_matrix,
			'grid_shape': self.grid.shape,
			'num_states': self.num_states,
			'obs_id_lookup': self.observation_id_lookup, # raw_obs -> obs_id
			'obs_id_reverse_lookup': self.observation_id_reverse_lookup, # obs_id ->  raw_obs
			'state_id_to_xy': self.transition_matrix_lookup, # state_id -> (x,y)
			'xy_to_state_id': self.transition_matrix_reverse_lookup, # (x,y) -> state_id
				}

	def calculate_action_transition_matrix(self):
		'''
		NOTE: Optional, feel free to ignore 
		
		Compute the action-conditioned transition model

			T_action[s, a, s'] = P(X_{t+1} = s' | X_t = s, A_t = a)

		If you are interested in experimenting with control/policies use this
		  method to return a numpy array (tensor) with shape: (N, len(Actions), N)
		  where for every (state s, action index a), T_action[s, a, :] is a valid
		  probability distribution over next states (non-negative and sums to 1).
		'''
		return None
	
	def calculate_transition_matrix(self):
		'''
		Compute the state transition probability matrix T
			where T[i,j] = P(X_t+1 = j | X_t = i)
		
		T is represented as a 2D numpy array where [i,j] corresponds to the 
			transition probability of the robot moving from state `i` to state `j` 
			T has shape (NxN) x (NxN); for example, in the case of N=20, T will be 400 x 400
			where N is the size of the square grid. 

		The transition probability could be uniform across all valid neighbors (free-cells) or
		  could include other stochastic environmental or robot dynamics. 

		If a free-cell is isolated (with no neighbors) then it is a self-loop and should have value 1.0

		Hints:
		- You may want to use the lookups/reverse-lookups to get state/observation indices
		- The Directions Enum provides your with direction definitions for convenience
		- 

		'''

		T = self.init_transition_matrix()

		## -------
		# TODO: Implement your own transition model 
	
		
		## -------

		# check transition matrix is valid
		row_sums = T.sum(axis=1)
		valid_states = np.any(T != 0, axis=1)
		assert np.allclose(row_sums[valid_states], 1), 'T contains rows that do not add to one'
		assert np.all(T>=0), 'T contains negative values'

		self.transition_matrix = T
	
	def calculate_observation_matrix(self):
		'''
		Compute the observation (emission) probability matrix O
			where O[i,o] = Prob( O_t = o | X_t = i )
		
		O is represented as a 2D numpy array where [i,o] corresponds to the 
			probability of observing `o` given the robot is in state `i` 
			O has shape (NxN) x (Z); (row x col)
			where N is the size of the square grid. 
			where Z is the unique observations possible in the grid from valid cells. 
			In our scenario we know the upper bound is 2^4=16; 
				since an observation is a binary vector of lenght four [N,E,S,W]
				so since each value can be {0,1}, then 2^4=16
			for example, in the case of N=20, and Z=16; O will be 400 x 16
			A more efficient way would be to only count observations possible in a given map,
			  such that a given map may have less than 16 actual observations. But since this
			  problem is small enough its ok to start with the upper-bound without incurring 
			  to much additional compute.

		The observation probability could be deterministic (no sensor noise) or stochastic, and could have aliasing.
		Aliasing is when the same observation may be possible from different states.

		Hints:
		- You already have access to self.all_free_cell_pos, a list of all free cell positions
		- You may want to use the lookups/reverse-lookups to get state/observation indices
		- the ground_truth_observation() method could be useful if you want to provide deterministic observations 
		'''

		O = self.init_observation_matrix()

		## -------
		# TODO: Implement your own observation model 
		
		
		## -------

		# check observation matrix is valid
		row_sums = O.sum(axis=1)
		valid_states = np.any(O != 0, axis=1)
		assert np.allclose(row_sums[valid_states], 1), 'O contains rows (states) that do not add to one'
		assert not np.allclose(row_sums[valid_states], 0) and not (np.all(O==0)), 'O contains rows (states) that have no observation. Note this error occurs because we define all possible observations upfront, even the zero (0,0,0,0) oberservation which is indexed in the matrix.'
 
		self.observation_matrix = O
	
	def init_agent_pos(self):
		'''choose a starting position uniformly at random from the available free-cells''' 
		return random.choice(self.all_free_cell_pos)

	def place_rectangle(self, grid, gridsize):
		'''given rectagle primities (varying widhts and heights) randomly place them in the grid'''
		while True:
			width = random.randint(1, 5)
			height = random.randint(1, 5)
			top_left_x = random.randint(0, gridsize - width)
			top_left_y = random.randint(0, gridsize - height)
			
			for x in range(top_left_x, top_left_x + width):
				for y in range(top_left_y, top_left_y + height):
					grid[x, y] = 1
			break

	def generate_obstacles(self):
		'''
		Generate a grid by randomly placing rectangles of varying sizes.
		we continue placing rectangles until a desired ratio of free-cells/obstacles is reached.
		create walls around the edges of the world.
		
		This function assumes square worlds.
		'''
		grid = np.zeros((self.N, self.N), dtype=int)
		gridsize = grid.shape[0] # assumes square grid
		total_cells = gridsize * gridsize
		num_obstacles = int(total_cells * self.obstacle_ratio)
		obstacle_count = 0
		
		# place random obstacles
		while obstacle_count < num_obstacles:
			current_obstacle_count = np.sum(grid)
			if current_obstacle_count >= num_obstacles:
				break
			self.place_rectangle(grid, gridsize)
			obstacle_count = np.sum(grid)
		
		# place the boundary walls
		for i in range(0,self.N):
			# top wall 
			grid[0,i] = 1
			# bottom wall
			grid[self.N-1,i] = 1
			# left wall 
			grid[i, 0] = 1
			# right wall 
			grid[i, self.N-1] = 1

		# return list of indices of obstacles in grid
		obstacles_xy = np.argwhere(grid == 1)
		free_xy = np.argwhere(grid == 0)
		return grid, obstacles_xy, free_xy

	def ground_truth_observation(self, cur_pos: np.ndarray):
		'''
		given a position on the map generate the ground-truth observation
		'''
		observable_obstacles = np.zeros_like(self.sensor_obs)
		# for each of the cardinal directions check if there's an obstacle
		for obs_idx, obs in enumerate(Observations):
			direction_i = obs.get_direction_xy()
			cell_at_dir = cur_pos + direction_i
			exist = np.any(np.all(self.all_obstacle_pos == cell_at_dir, axis=1))
			observable_obstacles[obs_idx] = 1 if exist else 0
		# convert to tuple (N, E, S, W), since numpy array are not hashable
		observable_obstacles_tuple = tuple(observable_obstacles.tolist())
		return observable_obstacles_tuple
	
	def sample_observation(self):
		'''
		sample an observation at the current agent position using observation matrix
		'''
		cur_state_id = self._get_state_id_from_pos(self.agent_pos)
		row = self.observation_matrix[cur_state_id]
		row_sum = row.sum()

		if not np.isclose(row_sum, 1.0):
			row = row / row_sum
		obs_id = np.random.choice(self.num_observations, p=row)

		# map obs_id -> raw (N,E,S,W) tuple
		return self.observation_id_reverse_lookup[obs_id]
	
	def motion_model(self, action: int):
		''' 
		Robot motion model
		
		Default behavior:
			Use the action-independent transition matrix T[s, s'], action is ignored by default
		Optional:
			If you implement self.transition_matrix_action (i.e. it is not None)
			the provided action is used to select T[s, a, :] when sampling the next state.
		'''
		cur_state_id = self._get_state_id_from_pos(self.agent_pos)
		# if action-conditioned transition is implemented
		if self.transition_matrix_action is not None:
			# map action name or enum to index in [0, len(Actions))
			if isinstance(action, str):
				action_enum = Actions[action]
			elif isinstance(action, Actions):
				action_enum = action
			else:
				raise ValueError('expected action to be either a string or Actions enum, received {}'.format(type(action)))
			action_idx = list(Actions).index(action_enum)
			row = self.transition_matrix_action[cur_state_id, action_idx]
		else:
			row = self.transition_matrix[cur_state_id] # default, use action-independent T

		row_sum = row.sum()
		if row_sum <= 0:
			raise ValueError('transition probabilities for state {} and action {} must sum to a positive value'.format(cur_state_id, action))
		if not np.isclose(row_sum, 1.0): # just in case there are numerical issues
			row = row / row_sum
		next_state_id = np.random.choice(self.num_states, p=row) # randomly choose based on prob
		next_xy = self.transition_matrix_lookup[next_state_id]
		self.agent_pos = np.array(next_xy, dtype=np.int32) # update robot position

	def __old_motion_model(self, action: int):
		''' 
		NOTE: This is an older implementation, deprecated now
		Robot motion model
		Robot is only allowed to move in free-cells; if an action is invalid we default to 'Stay'. 
		'''
		# map action (0...8) to the cell direction to move to
		direction_xy = Actions[action].get_direction_xy()

		# make sure the action is valid 
		next_pos = self.agent_pos + direction_xy
		# NOTE: Need to check with Prof. about the logic of moving when obstacles are not observable
		# assert not np.any(np.all(self.all_obstacle_pos == next_pos, axis=1)), 'Invalid action {}, there is an obstacle in position: {}'.format(Actions(action), next_pos)
		invalid_action = np.any(np.all(self.all_obstacle_pos == next_pos, axis=1))
		if invalid_action:
			return # do nothing
		else:
			self.agent_pos = next_pos

	def reset(self, seed=None, options=None):
		''' generate a new gridworld '''
		super().reset(seed=seed)

		# reset environment model/matrices and properties
		self.reset_env_model_properties()

		# generate obstacles 
		grid, obstacles_xy, free_xy = self.generate_obstacles()
		self.grid = grid
		self.all_obstacle_pos = obstacles_xy
		self.all_free_cell_pos = free_xy
		self.cur_step = 1
		self.prev_action = Actions.STAY 

		# construct environment models/matrices
		self.calculate_transition_matrix()
		self.calculate_observation_matrix()

		# optional, None by default, used to construct action-conditioned T
		self.transition_matrix_action = self.calculate_action_transition_matrix() 

		# place the agent randomly on the map
		self.agent_pos = self.init_agent_pos()

		# get current observation sample
		self.sensor_obs = self.sample_observation()
		
		# get any additional ground-truth or environment data 
		priviledged_info = self.get_priviledged_info()
		model_info = self.get_model_info()

		return self.sensor_obs, priviledged_info, model_info
	
	def step(self, action):
		''' step over the environment provided an action '''
		if action not in self.valid_actions:
			raise ValueError('expected input action of type {}; Received: {}'.format(Actions, action))
		
		# move robot
		self.motion_model(action)
		
		# get current observation sample
		self.sensor_obs  = self.sample_observation()
		
		# get any additional ground-truth or environment data 
		priviledged_info = self.get_priviledged_info()

		# this environment runs forever and has no rewards
		terminated = False
		truncated = False
		reward = 0
		self.cur_step += 1
		self.prev_action = action

		return self.sensor_obs, reward, terminated, truncated, priviledged_info
	
	'''
	Rendering methods
	'''

	def render(self, **kwargs):
		''' render the environment '''
		if self.render_mode == "rgb_array":
			return self._render_frame( **kwargs)
	
	def _render_frame(self, state_probabilities=None, suppress_plot_display=True):
		gridworld_render = self.render_gridworld(state_probabilities=state_probabilities)
		observation_render = self.render_observation()
		frame = self.get_frame(gridworld_render, observation_render, suppress_plot_display=suppress_plot_display)
		return frame
	
	def render_gridworld(self, state_probabilities=None, prob_threshold=1e-8):
		''' create image of the gridworld with optional state_probabilities'''

		cur_state_prob = None

		if state_probabilities is not None:
			if len(state_probabilities) == 0:
				raise ValueError('render expects `state_probabilities` to be a non-empty list or None.')
			# get current state probabilities
			cur_state_prob = state_probabilities[-1]

			# validate the type of state_probabilities
			if not isinstance(cur_state_prob, np.ndarray):
				raise TypeError('render expects `state_probabilities` to be a list of numpy arrays. Received type: {}'.format(type(cur_state_prob)))

			# validate the shape of state probabilties
			expected_shape = (self.num_states,)
			if cur_state_prob.shape != expected_shape:
				raise ValueError('invalid shape for `state_probabilities`. Received: {}, expected {}'.format(cur_state_prob.shape, expected_shape))

			# get minimum and maximum from history
			states_stacked = np.stack(state_probabilities)
			min_prob = np.min(states_stacked)
			max_prob = np.max(states_stacked)

		# canvas = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((255, 255, 255)) # white background
		pix_square_size = (self.window_size / self.N)  # size of a single grid square in pixels

		# draw the obstacles
		for obstacle in self.all_obstacle_pos:
			pygame.draw.rect(
				surface=canvas,
				color=(0, 0, 0),
				rect=pygame.Rect(
					pix_square_size * obstacle,
					(pix_square_size, pix_square_size),
				),
			)

		# draw state probabilities heatmap
		if cur_state_prob is not None:
			# set very low probabilities to zero (using threshold)
			cur_state_prob = np.where(cur_state_prob < prob_threshold, 0, cur_state_prob)
			# get custom color map
			cmap = self.custom_colormap()
			# normalize color map to the range of current probabilities
			norm = Normalize(vmin=min_prob, vmax=max_prob)
			# get corresponding (r,g,b)
			probs_to_rgba = cmap(norm(cur_state_prob))
			probs_to_rgb = (probs_to_rgba[:, :3] * 255).astype(int)

			# for each free-cell set the color given the probability
			for free_cell in self.all_free_cell_pos:
				cur_cell_as_tuple = tuple(free_cell.tolist())
				cell_id = self.transition_matrix_reverse_lookup[cur_cell_as_tuple]
				cell_rgb = tuple(probs_to_rgb[cell_id].tolist())

				pygame.draw.rect(
					surface=canvas,
					color=cell_rgb,
					rect=pygame.Rect(
						pix_square_size * free_cell,
						(pix_square_size, pix_square_size),
					),
				)

		# draw robot
		pygame.draw.circle(
			surface=canvas,
			color=(0, 0, 255),
			center=(self.agent_pos + 0.5) * pix_square_size,
			radius=pix_square_size / 3,
		)

		# draw grid lines
		for x in range(self.N + 1):
			pygame.draw.line(
				surface=canvas,
				color=(0, 0, 0),
				start_pos=(0, pix_square_size * x),
				end_pos=(self.window_size, pix_square_size * x),
				width=3,
			)
			pygame.draw.line(
				surface=canvas,
				color=(0, 0, 0),
				start_pos=(pix_square_size * x, 0),
				end_pos=(pix_square_size * x, self.window_size),
				width=3,
			)
		# we transpose from col,row -> row,col, and then flip the y-axis to match cartesian (x,y) coordinates
		return np.flipud(np.transpose(
			np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
		))

	
	def render_observation(self):
		''' create image of egocentric observations  '''
		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((255, 255, 255)) # white background
		observation_range = 3
		ego_agent_pos = np.zeros(shape=2)
		pix_square_size = (self.window_size / observation_range)  # size of a single grid square in pixels

		# collect position of obstacles in view
		observable_obstacles_xy = [Observations.get_direction_by_index(obs_idx).value for obs_idx, obs in enumerate(self.sensor_obs) if obs==1]
		
		# draw the obstacles
		for obstacle in observable_obstacles_xy:
			pygame.draw.rect(
				surface=canvas,
				color=(0, 0, 0),
				rect=pygame.Rect(
					pix_square_size * self.ego_to_allo_coords(obstacle, observation_range) ,
					(pix_square_size, pix_square_size),
				),
			)
		# draw non-sensor regions
		non_sensor_regions = [np.array(i) for i in [(1,1),(1,-1),(-1,-1),(-1,1)]]
		for cell in non_sensor_regions:
			pygame.draw.rect(
				surface=canvas,
				color=(200, 200, 200),
				rect=pygame.Rect(
					pix_square_size * self.ego_to_allo_coords(cell, observation_range) ,
					(pix_square_size, pix_square_size),
				),
			)

		# draw robot
		pygame.draw.circle(
			surface=canvas,
			color=(0, 0, 255),
			center=(self.ego_to_allo_coords(ego_agent_pos, observation_range) + 0.5) * pix_square_size,
			radius=pix_square_size / 3,
		)

		# draw grid lines
		for x in range(self.N + 1):
			pygame.draw.line(
				surface=canvas,
				color=(200, 200, 200),
				start_pos=(0, pix_square_size * x),
				end_pos=(self.window_size, pix_square_size * x),
				width=10,
			)
			pygame.draw.line(
				surface=canvas,
				color=(200, 200, 200),
				start_pos=(pix_square_size * x, 0),
				end_pos=(pix_square_size * x, self.window_size),
				width=10,
			)

		# we transpose from col,row -> row,col, and then flip the y-axis to match cartesian (x,y) coordinates
		return np.flipud(np.transpose(
			np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
		))
	
	def ego_to_allo_coords(self, ego_xy, window_size):
		''' egocentric to allocentric coordinate transformation '''
		window_center = window_size // 2
		window_center_xy = np.repeat(window_center, 2)
		allo_xy = window_center_xy + ego_xy
		return allo_xy

	def custom_colormap(self):
		''' create a custom color map for superimposed state probabilities '''
		# full range of colors from the default colormap
		plasma = plt.cm.plasma_r(np.linspace(0, 1, 256))
		# number of steps for the transition to white
		n_white = 1
		# RGB of the last color in colormap
		last_color = plasma[0, :3]  #
		# gradient of white to the last color
		white_transition = np.linspace([1, 1, 1], last_color, n_white)
		# combine colormap and transition to white
		colors = np.vstack([np.column_stack([white_transition, np.ones(n_white)]), plasma]) # include alpha using np.ones
		plasma_to_white = LinearSegmentedColormap.from_list('plasma_to_white', colors)
		return plasma_to_white

	def get_frame(self, gridworld_render, observation_render, suppress_plot_display=False):
		
		if suppress_plot_display:
			# prevent plot from being displayed on jupyter notebook
			plt.ioff()
		else: 
			plt.ion()

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12), gridspec_kw={'width_ratios': [3, 1]}) 

		# gridworld on left side subplot
		ax1.imshow(gridworld_render)
		ax1.axis('off')
		# ax1.set_title('t: {:3d} | prev-action: {}'.format(self.cur_step, self.prev_action), loc='left')
		ax1.set_title('t: {:3d}'.format(self.cur_step), loc='left')

		# sensor readings on right side subplot
		ax2.imshow(observation_render)
		ax2.axis('off')
		ax2.set_title('Sensor Observations')

		plt.subplots_adjust(wspace=0.08) 
		plt.axis('off')

		# save the plot to an in-memory buffer
		buf = BytesIO()
		fig.savefig(buf, format='png', bbox_inches='tight')
		buf.seek(0)  # rewind buffer to the beginning
		plt.close()
		return Image.open(buf) 
	
def save_gif(frames: List[Image.Image], save_to_path='figures/sample.gif'):
	if not isinstance(frames, list):
		raise TypeError('input frames must be a list of PIL Image objects')
		
	frames[0].save(
	save_to_path,
	save_all=True,
	append_images=frames[1:], # append all subsequent frames
	duration=200, # duration of each frame in milliseconds
	loop=0) # loop forever (set to 1 for one-time playback)

