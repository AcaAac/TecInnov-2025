'''
Self-Localization modeled as a Hidden Markov Model (HMM)
'''

import numpy as np
from typing import List, Optional, Tuple

def get_state_probabilities(observation: Tuple[int, int, int, int], 
							model_info: dict,
							observation_history: Optional[List[Tuple[int, int, int, int]]] = None, 
							dummy: bool = True) -> np.ndarray:
	'''
	TODO: implement your own state probabilities estimation, replace the placehold return with yours

	suggested inputs, feel free to modify/ignore:
		
		observation: tuple 
			- current sensor reading (N, E, S, W) as a binary tuple, i.e. (0, 1, 0, 1)

		model_info: dictionary of model information provided by the environment. Items inside:
			- transition_matrix: numpy array of shape (N*N, N*N)
				- state transition matrix P(x_t | x_{t-1})
			- observation_matrix: numpy array of shape (N*N, Z)	
				- observation likelihoods P(o_t | x_t)
			- grid_shape: (N,N) tuple of the gridworld dimensions
			- num_states: N*N total number of possible states (includes obstacles)
			- obs_id_lookup: map of raw obs tuple (N, E, S, W) to column index in observation matrix
			- obs_id_reverse_lookup: map of column index in observation matrix to raw obs tuple (N, E, S, W)
			- state_id_to_xy: map of state row/col id (equivalent to row/col since T is symmetric) to np.array(x,y)
			- xy_to_state_id: map of tuple (x,y) state to state row/col id (equivalent to row/col since T is symmetric). Note: tuple (x,y) is used as key because numpy arrays cannot be hashed for a lookup key.
		
		observation_history: list of past observations (tuples, same as observation)
		
		dummy: use placeholder dummy probabilities, set to False when testing your own implementation

	required outputs:
		numpy array of shape (N*N,), probability distribution over the robot position at time t
	'''
	if dummy: # baseline, do not modify, set dummy to False
		return _get_dummy_probs(observation,  model_info, observation_history)
	
	else:
		#return NotImplementedError('You must implement get_state_probabilities() \n You should remove/comment the `NotImplementedError` return when completed.')
		print('Filtering')

## -------------------------------------------------

def  _get_dummy_probs(priviledged_info, model_info, observation_history, 
					high_uncertainty_duration=0.5, sigma_range=(50.0,2.0), max_steps=100):
	'''
	simulate a dummy scenario with an arbitrary gaussian probability for converging to the true localization.
	dummy since it uses the "ground-truth" position. 

	dummy parameters: 
	- priviledged_info: get grount-truth agent position
	- model_info: contains information about environmental models
	- observation_history: used to compute current timestep to simulate ucnertainty decrease


	Internal parameters:
	- high_uncertainty_duration: how long to maintain a high standard deviation (std)
	- sigma_range: std bounds for scenario
	- max_steps: max number of steps in episode, used to decide how long to maintain high/low uncertainty

	'''    
	t = len(observation_history) # get a proxy for the current timestep from history length

	max_sigma, min_sigma = sigma_range
	# generate a sequence of decreasing sigmas
	sigmas = np.linspace(max_sigma, min_sigma, num=int(max_steps*high_uncertainty_duration))
	sigmas = np.concatenate([sigmas, min_sigma*np.ones(shape=(int(max_steps*(1-high_uncertainty_duration)),))])
	
	# compute new probabilities
	probs_t = _gaussian_2d_array(model_info['grid_shape'], sigmas[t], center=priviledged_info['agent_position']).flatten()

	return probs_t

def _gaussian_2d_array(shape, sigma=1.0, center=(0,0)):
	# create a meshgrid of x and y coordinates
	x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
	# calculate the distance a center provided
	dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	# generate the Gaussian values
	gaussian = np.exp(-(dist**2) / (2 * sigma**2))
	# normalize the values to sum to 1
	gaussian /= np.sum(gaussian)
	return gaussian