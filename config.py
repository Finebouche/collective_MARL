
run_config  = dict(
    # Sample YAML configuration for the tag continuous environment
    name= "tag_continuous",

    # Environment settings
    env =dict(
        num_preys= 50,
        num_predators= 1,
        stage_size= 40,
        episode_length= 300,
        preparation_length= 100,
        max_acceleration= 0.1,
        max_turn= 2.35,  # 3*pi/4 radians
        num_acceleration_levels= 10,
        num_turn_levels= 10,
        starving_penalty_for_predator= -1.0,
        eating_reward_for_predator= 1.0,
        surviving_reward_for_prey= 1.0,
        death_penalty_for_prey= -1.0,
        edge_hit_penalty= -0.1,
        end_of_game_penalty= -100.0,
        end_of_game_reward= 100.0,
        use_full_observation= False,
        eating_distance= 0.02,
        seed= 274880,
        env_backend= "numba",
    ),

    # Trainer settings
    trainer=dict(
        num_envs= 400, # number of environment replicas
        train_batch_size= 10000, # total batch size used for training per iteration (across all the environments)
        num_episodes= 700, # number of episodes to run the training for (can be arbitrarily high)
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
                module_name= "fully_connected", # model type
                class_name= "FullyConnected", # class type
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