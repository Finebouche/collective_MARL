import numpy as np

run_config  = dict(
    # Sample YAML configuration for the tag continuous environment
    name= "collective_emergence",

    # Environment settings
    env =dict(
        num_preys=80,
        num_predators=3,
        stage_size=50,
        episode_length=500,
        preparation_length=120,
        # Physics
        draging_force_coefficient = 0.5,
        contact_force_coefficient = 0.5,
        wall_contact_force_coefficient = 2,
        prey_size=0.2,
        predator_size=0.4,
        min_speed=0,
        max_speed=0.5,
        # Action parameters
        max_acceleration=0.2,
        min_acceleration=0,
        max_turn= np.pi/2,  # pi radians
        min_turn=- np.pi/2,  # pi radians
        num_acceleration_levels=10,
        num_turn_levels=10,
        # Reward parameters
        starving_penalty_for_predator=-0.0,
        eating_reward_for_predator=10000.0,
        surviving_reward_for_prey=.0,
        death_penalty_for_prey=-10000.0,
        edge_hit_penalty=-0,
        end_of_game_penalty=-0,
        end_of_game_reward=0,
        use_energy_cost=True,
        # Observation parameters
        use_full_observation=True, # Put False if not used
        max_seeing_angle= None,  # Put None if not used
        max_seeing_distance=None,  # Put None if not used
        num_other_agents_observed = None,  # Put None if not used
        use_time_in_observation=False,
        use_polar_coordinate=True,
        seed=None,
        env_backend="numba",
    ),

    # Trainer settings
    trainer=dict(
        num_envs= 500, # number of environment replicas
        train_batch_size= 1000, # total batch size used for training per iteration (across all the environments)
        num_episodes= 10000, # number of episodes to run the training for (can be arbitrarily high) 4m 30s for 10000
    ),
    # Policy network settings
    policy=dict( # list all the policies below
        prey=dict(
            to_train= True, # flag indicating whether the model needs to be trained
            algorithm= "PPO", # algorithm used to train the policy
            gamma= 0.98, # discount rate gamms
            lr= 0.001, # learning rate
            vf_loss_coeff= 1, # loss coefficient for the value function loss
            entropy_coeff= [[0, 0.5], [200000, 0.1], [2000000, 0.01]], # entropy coefficient (can be a list of lists)
            model=dict( # policy model settings
                type= "prey_policy",
                fc_dims= [64, 64, 64], # dimension(s) of the fully connected layers as a list
                model_ckpt_filepath= "", # filepath (used to restore a previously saved model)
            ),
        ),
        predator=dict(
            to_train= True,
            algorithm= "PPO",
            gamma= 0.98,
            lr= 0.001,
            vf_loss_coeff= 1,
            entropy_coeff= [[0, 0.5], [200000, 0.1], [2000000, 0.01]], # entropy coefficient (can be a list of lists)
            model=dict(
                type= "predator_policy",
                fc_dims= [64, 64, 64],
                model_ckpt_filepath= "",
            )
        )
    ),

    # Checkpoint saving setting
    saving=dict(
            metrics_log_freq= 100, # how often (in iterations) to print the metrics
            model_params_save_freq= 50000, # how often (in iterations) to save the model parameters
            basedir= "/tmp", # base folder used for saving
            name= "collective_v0",
            tag= "50preys_1predator",
            wandb=True,
            wandb_project="rl_project",
    )
)