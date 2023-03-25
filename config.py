import numpy as np

run_config  = dict(
    # Sample YAML configuration for the tag continuous environment
    name= "collective_emergence",

    # Environment settings
    env =dict(
        num_preys=50,
        num_predators=1,
        stage_size=50,
        episode_length=500,
        preparation_length=120,
        min_speed=0.2,
        max_speed=1,
        max_acceleration=0.1,
        min_acceleration=-0.1,
        max_turn=np.pi / 4,
        min_turn=-np.pi / 4,
        num_acceleration_levels=5,
        num_turn_levels=5,
        starving_penalty_for_predator=0,
        eating_reward_for_predator=1.0,
        surviving_reward_for_prey=0,
        death_penalty_for_prey=-1.0,
        edge_hit_penalty=-0.5,
        end_of_game_penalty=-10,
        end_of_game_reward=10,
        use_full_observation=False,
        max_seeing_angle=np.pi / 2,
        max_seeing_distance=8,
        num_other_agents_observed = None,
        eating_distance=0.02,
        seed=None,
        env_backend="numba",
    ),

    # Trainer settings
    trainer=dict(
        num_envs= 400, # number of environment replicas
        train_batch_size= 10000, # total batch size used for training per iteration (across all the environments)
        num_episodes= 1000, # number of episodes to run the training for (can be arbitrarily high)
    ),
    # Policy network settings
    policy=dict( # list all the policies below
        prey=dict(
            to_train= True, # flag indicating whether the model needs to be trained
            algorithm= "A2C", # algorithm used to train the policy
            gamma= 0.98, # discount rate gamms
            lr= 0.005, # learning rate
            vf_loss_coeff= 1, # loss coefficient for the value function loss
            entropy_coeff= [[0, 0.5], [2000000, 0.05]], # entropy coefficient (can be a list of lists)
            model=dict( # policy model settings
                type= "predator_policy",
                fc_dims= [256, 256], # dimension(s) of the fully connected layers as a list
                model_ckpt_filepath= "", # filepath (used to restore a previously saved model)
            ),
        ),
        predator=dict(
            to_train= True,
            algorithm= "A2C",
            gamma= 0.98,
            lr= 0.002,
            vf_loss_coeff= 1,
            model=dict(
                type= "fully_connected",
                fc_dims= [256, 256],
                model_ckpt_filepath= "",
            )
        )
    ),

    # Checkpoint saving setting
    saving=dict(
            metrics_log_freq= 100, # how often (in iterations) to print the metrics
            model_params_save_freq= 5000, # how often (in iterations) to save the model parameters
            basedir= "/tmp", # base folder used for saving
            name= "collective_v0",
            tag= "50preys_1predator",
    )
)