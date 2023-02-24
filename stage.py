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
_DIR = "direction"
_ACC = "acceleration"
_SIG = "still_in_the_game"
_DONE = "done"

class Environment(CUDAEnvironmentContext):

    def __init__(self,
                 num_preys=50,
                 num_predators=1,
                 stage_size=100.0,
                 episode_length=240,
                 preparation_length=120,
                 starting_location_x=None,
                 starting_location_y=None,
                 max_speed=1.0,
                 max_acceleration=1.0,
                 min_acceleration=-1.0,
                 max_turn=np.pi / 2,
                 min_turn=-np.pi / 2,
                 num_acceleration_levels=10,
                 num_turn_levels=10,
                 edge_hit_penalty=-10,
                 tag_reward_for_tagger=10.0,
                 tag_penalty_for_runner=-10.0,
                 end_of_game_penalty=-1,
                 end_of_game_reward=1,
                 use_full_observation=True,
                 runner_exits_game_after_tagged=True,
                 tagging_distance=0.02,
                 seed=None,
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

        self.agent_type = {}    # 0 for preys, 1 for predators
        self.predators = {}     # predator ids
        self.preys = {}         # prey ids
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

        # Set the starting directions
        self.starting_directions = self.np_random.choice(
            [0, np.pi / 2, np.pi, np.pi * 3 / 2], self.num_agents, replace=True
        )

        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

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

        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (len(self.acceleration_actions), len(self.turn_actions))
            )
            for agent_id in range(self.num_agents)
        }

        # OBSERVATION SPACE
        self.observation_space = None  # Note: this will be set via the env_wrapper

        # Used in generate_observation()
        self.init_obs = None  # Will be set later in generate_observation()

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

    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        """
        obs = {}

        normalized_global_obs = None
        for feature in [
            (_LOC_X, self.grid_diagonal),
            (_LOC_Y, self.grid_diagonal),
            (_SP, self.max_speed + self.eps),
            (_ACC, self.max_speed + self.eps),
            (_DIR, 2 * np.pi),
        ]:
            if normalized_global_obs is None:
                normalized_global_obs = (
                    self.global_state[feature[0]][self.timestep] / feature[1]
                )
            else:
                normalized_global_obs = np.vstack(
                    (
                        normalized_global_obs,
                        self.global_state[feature[0]][self.timestep] / feature[1],
                    )
                )
        agent_types = np.array(
            [self.agent_type[agent_id] for agent_id in range(self.num_agents)]
        )
        time = np.array([float(self.timestep) / self.episode_length])

        if self.use_full_observation:
            for agent_id in range(self.num_agents):
                # Initialize obs
                obs[agent_id] = np.concatenate(
                    [
                        np.vstack(
                            (
                                np.zeros_like(normalized_global_obs),
                                agent_types,
                                self.still_in_the_game,
                            )
                        )[
                            :,
                            [idx for idx in range(self.num_agents) if idx != agent_id],
                        ].reshape(
                            -1
                        ),  # filter out the obs for the current agent
                        np.array([0.0]),
                    ]
                )
                # Set obs for agents still in the game
                if self.still_in_the_game[agent_id]:
                    obs[agent_id] = np.concatenate(
                        [
                            np.vstack(
                                (
                                    normalized_global_obs
                                    - normalized_global_obs[:, agent_id].reshape(-1, 1),
                                    agent_types,
                                    self.still_in_the_game,
                                )
                            )[
                                :,
                                [
                                    idx
                                    for idx in range(self.num_agents)
                                    if idx != agent_id
                                ],
                            ].reshape(
                                -1
                            ),  # filter out the obs for the current agent
                            time,
                        ]
                    )
        else:  # use partial observation
            for agent_id in range(self.num_agents):
                if self.timestep == 0:
                    # Set obs to all zeros
                    obs_global_states = np.zeros(
                        (
                            normalized_global_obs.shape[0],
                            self.num_other_agents_observed,
                        )
                    )
                    obs_agent_types = np.zeros(self.num_other_agents_observed)
                    obs_still_in_the_game = np.zeros(self.num_other_agents_observed)

                    # Form the observation
                    self.init_obs = np.concatenate(
                        [
                            np.vstack(
                                (
                                    obs_global_states,
                                    obs_agent_types,
                                    obs_still_in_the_game,
                                )
                            ).reshape(-1),
                            np.array([0.0]),  # time
                        ]
                    )

                # Initialize obs to all zeros
                obs[agent_id] = self.init_obs

                # Set obs for agents still in the game
                if self.still_in_the_game[agent_id]:
                    nearest_neighbor_ids = self.k_nearest_neighbors(
                        agent_id, k=self.num_other_agents_observed
                    )
                    # For the case when the number of remaining agent ids is fewer
                    # than self.num_other_agents_observed (because agents have exited
                    # the game), we also need to pad obs wih zeros
                    obs_global_states = np.hstack(
                        (
                            normalized_global_obs[:, nearest_neighbor_ids]
                            - normalized_global_obs[:, agent_id].reshape(-1, 1),
                            np.zeros(
                                (
                                    normalized_global_obs.shape[0],
                                    self.num_other_agents_observed
                                    - len(nearest_neighbor_ids),
                                )
                            ),
                        )
                    )
                    obs_agent_types = np.hstack(
                        (
                            agent_types[nearest_neighbor_ids],
                            np.zeros(
                                (
                                    self.num_other_agents_observed
                                    - len(nearest_neighbor_ids)
                                )
                            ),
                        )
                    )
                    obs_still_in_the_game = (
                        np.hstack(
                            (
                                self.still_in_the_game[nearest_neighbor_ids],
                                np.zeros(
                                    (
                                        self.num_other_agents_observed
                                        - len(nearest_neighbor_ids)
                                    )
                                ),
                            )
                        ),
                    )

                    # Form the observation
                    obs[agent_id] = np.concatenate(
                        [
                            np.vstack(
                                (
                                    obs_global_states,
                                    obs_agent_types,
                                    obs_still_in_the_game,
                                )
                            ).reshape(-1),
                            time,
                        ]
                    )

        return obs

    def compute_reward(self):
        """
        Compute and return the rewards for each agent.
        """
        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        taggers_list = sorted(self.taggers)

        # At least one runner present
        if self.num_runners > 0:
            runners_list = sorted(self.runners)
            runner_locations_x = self.global_state[_LOC_X][self.timestep][runners_list]
            tagger_locations_x = self.global_state[_LOC_X][self.timestep][taggers_list]

            runner_locations_y = self.global_state[_LOC_Y][self.timestep][runners_list]
            tagger_locations_y = self.global_state[_LOC_Y][self.timestep][taggers_list]

            runners_to_taggers_distances = np.sqrt(
                (
                    np.repeat(runner_locations_x, self.num_taggers)
                    - np.tile(tagger_locations_x, self.num_runners)
                )
                ** 2
                + (
                    np.repeat(runner_locations_y, self.num_taggers)
                    - np.tile(tagger_locations_y, self.num_runners)
                )
                ** 2
            ).reshape(self.num_runners, self.num_taggers)

            min_runners_to_taggers_distances = np.min(
                runners_to_taggers_distances, axis=1
            )
            argmin_runners_to_taggers_distances = np.argmin(
                runners_to_taggers_distances, axis=1
            )
            nearest_tagger_ids = [
                taggers_list[idx] for idx in argmin_runners_to_taggers_distances
            ]

        # Rewards
        # Add edge hit reward penalty and the step rewards/ penalties
        for agent_id in range(self.num_agents):
            if self.still_in_the_game[agent_id]:
                rew[agent_id] += self.edge_hit_reward_penalty[agent_id]
                rew[agent_id] += self.step_rewards[agent_id]

        for idx, runner_id in enumerate(runners_list):
            if min_runners_to_taggers_distances[idx] < self.distance_margin_for_reward:

                # the runner is tagged!
                rew[runner_id] += self.tag_penalty_for_runner
                rew[nearest_tagger_ids[idx]] += self.tag_reward_for_tagger

                if self.runner_exits_game_after_tagged:
                    # Remove runner from game
                    self.still_in_the_game[runner_id] = 0
                    del self.runners[runner_id]
                    self.num_runners -= 1
                    self.global_state[_SIG][self.timestep :, runner_id] = 0

        if self.timestep == self.episode_length:
            for runner_id in self.runners:
                rew[runner_id] += self.end_of_game_reward_for_runner

        return rew

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _SP, _DIR, _ACC]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )
        data_dict.add_data(
            name="agent_types",
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
        )
        data_dict.add_data(
            name="num_runners", data=self.num_runners, save_copy_and_apply_at_reset=True
        )
        data_dict.add_data(
            name="num_other_agents_observed", data=self.num_other_agents_observed
        )
        data_dict.add_data(name="grid_length", data=self.grid_length)
        data_dict.add_data(
            name="edge_hit_reward_penalty",
            data=self.edge_hit_reward_penalty,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="step_rewards",
            data=self.step_rewards,
        )
        data_dict.add_data(name="edge_hit_penalty", data=self.edge_hit_penalty)
        data_dict.add_data(name="max_speed", data=self.max_speed)
        data_dict.add_data(name="acceleration_actions", data=self.acceleration_actions)
        data_dict.add_data(name="turn_actions", data=self.turn_actions)
        data_dict.add_data(name="skill_levels", data=self.skill_levels)
        data_dict.add_data(name="use_full_observation", data=self.use_full_observation)
        data_dict.add_data(
            name="distance_margin_for_reward", data=self.distance_margin_for_reward
        )
        data_dict.add_data(
            name="tag_reward_for_tagger", data=self.tag_reward_for_tagger
        )
        data_dict.add_data(
            name="tag_penalty_for_runner", data=self.tag_penalty_for_runner
        )
        data_dict.add_data(
            name="end_of_game_reward_for_runner",
            data=self.end_of_game_reward_for_runner,
        )
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
        data_dict.add_data(
            name="runner_exits_game_after_tagged",
            data=self.runner_exits_game_after_tagged,
        )
        data_dict.add_data(
            name="still_in_the_game",
            data=self.still_in_the_game,
            save_copy_and_apply_at_reset=True,
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
        if not self.env_backend == "cpu":
            # CUDA version of step()
            # This subsumes update_state(), generate_observation(),
            # and compute_reward()

            args = [
                _LOC_X,
                _LOC_Y,
                _SP,
                _DIR,
                _ACC,
                "agent_types",
                "edge_hit_reward_penalty",
                "edge_hit_penalty",
                "grid_length",
                "acceleration_actions",
                "turn_actions",
                "max_speed",
                "num_other_agents_observed",
                "skill_levels",
                "runner_exits_game_after_tagged",
                "still_in_the_game",
                "use_full_observation",
                _OBSERVATIONS,
                _ACTIONS,
                "neighbor_distances",
                "neighbor_ids_sorted_by_distance",
                "nearest_neighbor_ids",
                _REWARDS,
                "step_rewards",
                "num_runners",
                "distance_margin_for_reward",
                "tag_reward_for_tagger",
                "tag_penalty_for_runner",
                "end_of_game_reward_for_runner",
                "_done_",
                "_timestep_",
                ("n_agents", "meta"),
                ("episode_length", "meta"),
            ]

            if self.env_backend == "pycuda":
                self.cuda_step(
                    *self.cuda_step_function_feed(args),
                    block=self.cuda_function_manager.block,
                    grid=self.cuda_function_manager.grid,
                )
            elif self.env_backend == "numba":
                self.cuda_step[
                    self.cuda_function_manager.grid, self.cuda_function_manager.block
                ](*self.cuda_step_function_feed(args))
            result = None  # do not return anything
        else:
            assert isinstance(actions, dict)
            assert len(actions) == self.num_agents

            acceleration_action_ids = [
                actions[agent_id][0] for agent_id in range(self.num_agents)
            ]
            turn_action_ids = [
                actions[agent_id][1] for agent_id in range(self.num_agents)
            ]

            assert all(
                0 <= acc <= self.num_acceleration_levels
                for acc in acceleration_action_ids
            )
            assert all(0 <= turn <= self.num_turn_levels for turn in turn_action_ids)

            delta_accelerations = self.acceleration_actions[acceleration_action_ids]
            delta_turns = self.turn_actions[turn_action_ids]

            # Update state and generate observation
            self.update_state(delta_accelerations, delta_turns)
            if self.env_backend == "cpu":
                obs = self.generate_observation()

            # Compute rewards and done
            rew = self.compute_reward()
            done = {
                "__all__": (self.timestep >= self.episode_length)
                or (self.num_runners == 0)
            }
            info = {}

            result = obs, rew, done, info
        return result