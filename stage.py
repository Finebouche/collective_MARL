import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

class Environment(CUDAEnvironmentContext):

    def __init__(num_preys = 50, 
                 num_predator = 1, 
                 stage_size = 100.0,
                 episod_length = 240,
                 preparation_length = 120,
                 starting_location_x=None,
                 starting_location_y=None,
                 max_speed=1.0,
                 max_acceleration=1.0,
                 min_acceleration=-1.0,
                 max_turn=np.pi / 2,
                 min_turn=-np.pi / 2,
                 num_acceleration_levels=10,
                 num_turn_levels=10,
                 edge_hit_penalty=-1.0,
                ):
        super().__init__()

        # ENVIRONMENT
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)
        
        # Seeding
        self.np_random = np.random
        if seed is not None:
        self.seed(seed)
        
        # Length and preparation
        assert episode_length > 0
        self.episode_length = episode_length
        self.preparation_length = preparation_length
        
        # Square 2D grid
        assert grid_length > 0
        self.grid_length = self.float_dtype(grid_length)
        self.grid_diagonal = self.grid_length * np.sqrt(2)


        # AGENTS
        assert num_preys > 0
        self.num_preys = num_preys
        assert num_predators > 0
        self.num_predators = num_predators
        self.num_agents = self.num_preys + self.num_predators
        
        # Initializing the ids of predators and preys
        # predators' ids
        predators_ids = self.np_random.choice(
            np.arange(self.num_agents), self.num_predators, replace=False
        )

        self.agent_type = {}
        self.predators = {}
        self.preys = {}
        for agent_id in range(self.num_agents):
            if agent_id in set(predators_ids):
                self.agent_type[agent_id] = 1  # Predators
                self.predators[agent_id] = True
            else:
                self.agent_type[agent_id] = 0  # Preys
                self.preys[agent_id] = True

                
        # Set the starting positions
        if starting_location_x is None:
            assert starting_location_y is None

            starting_location_x = self.grid_length * self.np_random.rand(
                self.num_agents
            )
            starting_location_y = self.grid_length * self.np_random.rand(
                self.num_agents
            )
        else:
            assert len(starting_location_x) == self.num_agents
            assert len(starting_location_y) == self.num_agents
            
        self.starting_location_x = starting_location_x
        self.starting_location_y = starting_location_y
        
        self.starting_directions = self.np_random.choice(
                [0, np.pi / 2, np.pi, np.pi * 3 / 2], self.num_agents, replace=True
            )
        

        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0

        # Set the max speed level
        self.max_speed = self.float_dtype(max_speed)
        
        # ACTION SPACE
        # The num_acceleration and num_turn levels refer to the number of
        # uniformly-spaced levels between (min_acceleration and max_acceleration)
        # and (min_turn and max_turn), respectively.
        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0
        self.num_acceleration_levels = num_acceleration_levels
        self.num_turn_levels = num_turn_levels
        self.max_acceleration = self.float_dtype(max_acceleration)
        self.min_acceleration = self.float_dtype(min_acceleration)

        self.max_turn = self.float_dtype(max_turn)
        self.min_turn = self.float_dtype(min_turn)

        # Acceleration actions
        self.acceleration_actions = np.linspace(
            self.min_acceleration, self.max_acceleration, self.num_acceleration_levels
        )
        # Add action 0 - this will be the no-op, or 0 acceleration
        self.acceleration_actions = np.insert(self.acceleration_actions, 0, 0).astype(
            self.float_dtype
        )

        # Turn actions
        self.turn_actions = np.linspace(
            self.min_turn, self.max_turn, self.num_turn_levels
        )
        # Add action 0 - this will be the no-op, or 0 turn
        self.turn_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)

        
        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None
        
        # OBSERVATION AND ACTIONS SPACE
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (len(self.acceleration_actions), len(self.turn_actions))
            )
            for agent_id in range(self.num_agents)
        }
        
        # Used in generate_observation()
        self.init_obs = None  # Will be set later in generate_observation()
        assert num_other_agents_observed <= self.num_agents
        self.num_other_agents_observed = num_other_agents_observed

        # Distance margin between agents for eating
        # If a predator is closer than this to a prey,
        # the predator eats the prey
        assert 0 <= eating_distance <= 1
        self.distance_margin_for_reward = (tagging_distance * self.grid_length).astype(
            self.float_dtype
        )
        
    name = "SimulationEnvironment"
    
    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def set_global_state(self, key=None, value=None, t=None, dtype=None):
        """
        Set the global state for a specified key, value and timestep.
        Note: for a new key, initialize global state to all zeros.
        """
        assert key is not None
        if dtype is None:
            dtype = self.float_dtype

        # If no values are passed, set everything to zeros.
        if key not in self.global_state:
            self.global_state[key] = np.zeros(
                (self.episode_length + 1, self.num_agents), dtype=dtype
            )

        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == self.global_state[key].shape[1]

            self.global_state[key][t] = value


