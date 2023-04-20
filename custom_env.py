import copy
import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_SP = "speed"
_ORI = "orientation"
_ACC = "acceleration"
_SIG = "still_in_the_game"
_DONE = "done"


class CustomEnv(CUDAEnvironmentContext):

    def __init__(self,
                 num_preys=50,
                 num_predators=1,
                 stage_size=100.0,
                 episode_length=240,
                 preparation_length=120,
                 starting_location_x=None,
                 starting_location_y=None,
                 use_physics = False,
                 eating_distance=0.02,
                 min_speed=0.2,
                 max_speed=0.5,
                 max_acceleration=0.5,
                 min_acceleration=-0.5,
                 max_turn=np.pi / 4,
                 min_turn=-np.pi / 4,
                 num_acceleration_levels=5,
                 num_turn_levels=5,
                 starving_penalty_for_predator=-1.0,
                 eating_reward_for_predator=1.0,
                 surviving_reward_for_prey=1.0,
                 death_penalty_for_prey=-1.0,
                 edge_hit_penalty=-0.1,
                 end_of_game_penalty=-10,
                 end_of_game_reward=10,
                 use_full_observation=True,
                 max_seeing_angle=None,
                 max_seeing_distance=None,
                 num_other_agents_observed = None,
                 use_time_in_observation=True,
                 use_polar_coordinate=False,
                 seed=None,
                 env_backend="numba",
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
        assert stage_size > 0
        self.stage_size = self.float_dtype(stage_size)
        self.grid_diagonal = self.stage_size * np.sqrt(2)

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

        self.agent_type = {}  # 0 for preys, 1 for predators
        self.predators = {}  # predator ids
        self.preys = {}  # prey ids
        for agent_id in range(self.num_agents):
            if agent_id in set(predators_ids):
                self.agent_type[agent_id] = 1  # Predators
                self.predators[agent_id] = True
            else:
                self.agent_type[agent_id] = 0  # Preys
                self.preys[agent_id] = True

        # Set the starting positions
        # If no starting positions are provided, then randomly initialize them
        if starting_location_x is None:
            assert starting_location_y is None
            starting_location_x = self.stage_size * self.np_random.rand(
                self.num_agents
            )
            starting_location_y = self.stage_size * self.np_random.rand(
                self.num_agents
            )
        else:
            assert len(starting_location_x) == self.num_agents
            assert len(starting_location_y) == self.num_agents
        self.starting_location_x = starting_location_x
        self.starting_location_y = starting_location_y

        # Set the starting directions
        self.starting_directions = self.np_random.choice(
            [0, np.pi / 2, np.pi, np.pi * 3 / 2], self.num_agents, replace=True
        )

        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        # Set the max speed level
        self.max_speed = self.float_dtype(max_speed)
        self.min_speed = self.float_dtype(min_speed)

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

        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (len(self.acceleration_actions), len(self.turn_actions))
            )
            for agent_id in range(self.num_agents)
        }

        # OBSERVATION SPACE
        if sum(var for var in [use_full_observation, num_other_agents_observed is not None, (max_seeing_angle is not None and max_seeing_distance is not None)]) != 1:
            raise ValueError("Only one of use_full_observation, num_other_agents_observed, and max_seeing_angle should be set.")


        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.use_full_observation = use_full_observation
        self.num_other_agents_observed = self.num_agents if num_other_agents_observed is None else num_other_agents_observed
        self.max_seeing_angle = stage_size/np.sqrt(2) if max_seeing_angle is None else max_seeing_angle
        self.max_seeing_distance = np.pi if max_seeing_distance is None else max_seeing_distance
        
        self.use_time_in_observation = use_time_in_observation
        self.use_polar_coordinate = use_polar_coordinate
        
        # Used in generate_observation()
        self.init_obs = None  # Will be set later in generate_observation()

        # Distance margin between agents for eating
        # If a predator is closer than this to a prey,
        # the predator eats the prey
        assert 0 <= eating_distance <= 1
        self.eating_distance = (eating_distance * self.stage_size).astype(self.float_dtype)

        # REWARDS
        self.starving_penalty_for_predator = starving_penalty_for_predator
        self.eating_reward_for_predator = eating_reward_for_predator
        self.surviving_reward_for_prey = surviving_reward_for_prey
        self.death_penalty_for_prey = death_penalty_for_prey
        self.edge_hit_penalty = edge_hit_penalty
        self.end_of_game_penalty = end_of_game_penalty
        self.end_of_game_reward = end_of_game_reward

        # Copy preys dict for applying at reset
        self.preys_at_reset = copy.deepcopy(self.preys)

        # These will also be set via the env_wrapper
        self.env_backend = env_backend

    name = "CustomEnv"

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


    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        """
        obs = {}

        normalized_global_obs = None
        # Normalize global states
        # Global states is a dictionary of numpy arrays
        # Each key is a feature, and each value is a numpy array of shape (episode_length, num_agents)
        for feature in [
            (_SP, self.max_speed + self.eps),
            (_ORI, 2 * np.pi),
            (_ACC, self.max_speed + self.eps),
            (_LOC_X, self.grid_diagonal),
            (_LOC_Y, self.grid_diagonal),
        ]:
            if normalized_global_obs is None:
                normalized_global_obs = self.global_state[feature[0]][self.timestep] / feature[1]
            else:
                normalized_global_obs = np.vstack(
                    (normalized_global_obs, self.global_state[feature[0]][self.timestep] / feature[1],)
                )

        # Agent types
        agent_types = np.array([self.agent_type[agent_id] for agent_id in range(self.num_agents)])
        # Time to indicate that the agent is still in the game
        time = np.array([float(self.timestep) / self.episode_length])

        if self.timestep != 0:
            raise ValueError("This function should be call only for timestep 0.")
            
        for agent_id in range(self.num_agents):
            if self.still_in_the_game[agent_id] and self.use_full_observation:
                obs[agent_id] = np.concatenate(
                    [
                        np.vstack((
                            normalized_global_obs - normalized_global_obs[:, agent_id].reshape(-1, 1),
                            agent_types,
                            self.still_in_the_game,
                            # is_visible
                        ))[:, [idx for idx in range(self.num_agents) if idx != agent_id], ].reshape(-1),
                        # filter out the obs for the current agent
                        time,
                    ]
                )
            else:
                # Set to 0
                # obs = [global_obs, agent_types, still_in_the_game, 0]
                obs[agent_id] = np.concatenate(
                    [
                        np.vstack((
                            np.zeros_like(normalized_global_obs),
                            agent_types,
                            self.still_in_the_game,
                        ))[:, [idx for idx in range(self.num_agents) if idx != agent_id], ].reshape(-1),
                        # filter out the obs for the current agent
                        np.array([0.0]),
                    ]
                )
        return obs


    def reset(self):
        """
        Env reset().
        """
        # Reset time to the beginning
        self.timestep = 0
        
        # Re-initialize the global state
        self.global_state = {}
        
        # Array to keep track of the agents that are still in play
        self.still_in_the_game = np.ones(self.num_agents, dtype=self.int_dtype)

        # Initialize global state for "still_in_the_game" to all ones
        self.global_state[_SIG] = np.ones((self.episode_length + 1, self.num_agents), dtype=self.int_dtype)
        self.set_global_state(key=_SP, value=self.starting_speeds, t=self.timestep)
        self.set_global_state(key=_ORI, value=self.starting_directions, t=self.timestep)
        self.set_global_state(key=_ACC, value=self.starting_directions, t=self.timestep)
        self.set_global_state(key=_LOC_X, value=self.starting_location_x, t=self.timestep)
        self.set_global_state(key=_LOC_Y, value=self.starting_location_y, t=self.timestep)        
        
        # Penalty for hitting the edges
        self.edge_hit_reward_penalty = np.zeros(self.num_agents, dtype=self.float_dtype)

        # Reinitialize some variables that may have changed during previous episode
        self.preys = copy.deepcopy(self.preys_at_reset)
        self.num_preys = len(self.preys)

        return self.generate_observation()


class CUDACustomEnv(CustomEnv, CUDAEnvironmentContext):
    """
    CUDA version of the CustomEnv environment.
    Note: this class subclasses the Python environment class CustomEnv,
    and also the  CUDAEnvironmentContext
    """

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _SP, _ORI, _ACC]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )

        data_dict.add_data(
            name="agent_types",
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
        )
        data_dict.add_data_list(
            [
                ("stage_size", self.stage_size),
                ("acceleration_actions", self.acceleration_actions),
                ("turn_actions", self.turn_actions),
                ("min_speed", self.min_speed),
                ("max_speed", self.max_speed),
                ("max_acceleration", self.max_acceleration),
                ("min_acceleration", self.min_acceleration),
                ("max_turn", self.max_turn),
                ("min_turn", self.min_turn),
            ]
        )
        data_dict.add_data(
            name="still_in_the_game",
            data=self.still_in_the_game,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="use_full_observation", data=self.use_full_observation)
        data_dict.add_data(name="num_other_agents_observed", data=self.num_other_agents_observed)
        data_dict.add_data(
            name="neighbor_distances",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.float32),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="neighbor_ids_sorted_by_distance",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="nearest_neighbor_ids",
            data=np.zeros(
                (self.num_agents, self.num_other_agents_observed), dtype=np.int32
            ),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="max_seeing_angle", data=self.max_seeing_angle)
        data_dict.add_data(name="max_seeing_distance", data=self.max_seeing_distance)
        data_dict.add_data(name="use_time_in_observation", data=self.use_time_in_observation)
        data_dict.add_data(name="use_polar_coordinate", data=self.use_polar_coordinate)

        
        # _OBSERVATIONS,
        # _ACTIONS,
        data_dict.add_data(
            name="edge_hit_reward_penalty",
            data=self.edge_hit_reward_penalty,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="num_preys", data=self.num_preys, save_copy_and_apply_at_reset=True
        )
        data_dict.add_data(
            name="num_predators", data=self.num_predators, save_copy_and_apply_at_reset=True
        )
        # _REWARDS,
        data_dict.add_data_list(
            [
                ("edge_hit_penalty", self.edge_hit_penalty),
                ("starving_penalty_for_predator", self.starving_penalty_for_predator),
                ("eating_reward_for_predator", self.eating_reward_for_predator),
                ("surviving_reward_for_prey", self.surviving_reward_for_prey),
                ("death_penalty_for_prey", self.death_penalty_for_prey),
                ("end_of_game_penalty", self.end_of_game_penalty),
                ("end_of_game_reward", self.end_of_game_reward),
            ]
        )
        data_dict.add_data(
            name="eating_distance", data=self.eating_distance
        )

        return data_dict

    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        return tensor_dict

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        # CUDA version of step()
        # This subsumes update_state(), generate_observation(),
        # and compute_reward()
        args = [
            _LOC_X,
            _LOC_Y,
            _SP,
            _ORI,
            _ACC,
            "agent_types",
            "stage_size",
            "acceleration_actions",
            "turn_actions",
            "min_speed",
            "max_speed",
            "max_acceleration",
            "min_acceleration",
            "max_turn",
            "min_turn",
            "still_in_the_game",
            _OBSERVATIONS,
            "use_full_observation",
            "num_other_agents_observed",
            "neighbor_distances",
            "neighbor_ids_sorted_by_distance",
            "nearest_neighbor_ids",
            "max_seeing_angle",
            "max_seeing_distance",
            "use_time_in_observation",
            "use_polar_coordinate",
            _ACTIONS,
            "edge_hit_reward_penalty",
            _REWARDS,
            "num_preys",
            "num_predators",
            "edge_hit_penalty",
            "starving_penalty_for_predator",
            "eating_reward_for_predator",
            "surviving_reward_for_prey",
            "death_penalty_for_prey",
            "end_of_game_penalty",
            "end_of_game_reward",
            "eating_distance",
            "_done_",
            "_timestep_",
            ("n_agents", "meta"),
            ("episode_length", "meta"),
        ]

        if self.env_backend == "numba":
            grid = self.cuda_function_manager.grid
            block = self.cuda_function_manager.block

            self.cuda_step[grid, block](
                *self.cuda_step_function_feed(args)
            )
        else:
            raise Exception("CUDACustomEnv expects env_backend = 'numba' ")