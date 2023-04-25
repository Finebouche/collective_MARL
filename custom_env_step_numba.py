import math

import numba.cuda as numba_driver
from numba import float32, int32, boolean, types

kTwoPi = 6.283185308
kEpsilon = 1.0e-10


# Device helper function to compute distances between two agents
@numba_driver.jit((float32[:, ::1],
                   float32[:, ::1],
                   int32,
                   int32,
                   int32),
                  device=True,
                  inline=True)
def ComputeDistance(
        loc_x_arr, loc_y_arr, kThisAgentId1, kThisAgentId2, kEnvId
):
    return math.sqrt(
        ((loc_x_arr[kEnvId, kThisAgentId1] - loc_x_arr[kEnvId, kThisAgentId2]) ** 2)
        + ((loc_y_arr[kEnvId, kThisAgentId1] - loc_y_arr[kEnvId, kThisAgentId2]) ** 2)
    )

# Device helper function to compute angle between two agents
@numba_driver.jit((float32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   int32,
                   int32,
                   int32),
                  device=True,
                  inline=True)
def ComputeAngle(
        loc_x_arr, loc_y_arr, orientation_arr, kThisAgentId1, kThisAgentId2, kEnvId
):
    return math.degrees(
        math.atan2(
            loc_y_arr[kEnvId, kThisAgentId1] - loc_y_arr[kEnvId, kThisAgentId2], 
            loc_x_arr[kEnvId, kThisAgentId1] - loc_x_arr[kEnvId, kThisAgentId2]
        ) - orientation_arr[kEnvId, kThisAgentId1]
    )

# Device helper function to compute angle between two agents
@numba_driver.jit((float32[:, :, ::1],
                   int32,
                   int32,
                   int32,
                   int32,
                   int32,
                   float32[:, ::1],
                   float32[:, ::1],
                   int32,
                   float32[:, ::1],
                   int32,
                   float32[:, ::1],
                   boolean,
                  ),
                  device=True,
                  inline=True)
def UpdateObservation(
    obs_arr,
    kEnvId, 
    kThisAgentId, 
    kNumAgents, 
    index, 
    other_agent_id, 
    loc_x_arr,
    loc_y_arr,
    kStageSize,
    speed_arr,
    kMaxSpeed,
    orientation_arr,
    kUsePolarCoordinate,
):
    # update relative normalized speed of the other agent
    obs_arr[kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index] = float(
        speed_arr[kEnvId, other_agent_id] - speed_arr[kEnvId, kThisAgentId]
    ) / (kMaxSpeed + kEpsilon)
    # update relative normalized orientation of the other agent
    obs_arr[kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index] = float(
        orientation_arr[kEnvId, other_agent_id] - orientation_arr[kEnvId, kThisAgentId]
    ) / kTwoPi
    if kUsePolarCoordinate: 
        ditance = ComputeDistance(loc_x_arr, loc_y_arr, kThisAgentId, other_agent_id, kEnvId)
        angle = ComputeAngle(loc_x_arr, loc_y_arr, orientation_arr, kThisAgentId, other_agent_id, kEnvId)
        # update relative distance of the other agent
        obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = ditance / (math.sqrt(2.0) * kStageSize)
        # update relative direction of the other agent
        obs_arr[kEnvId, kThisAgentId, 5 * (kNumAgents - 1) + index] = angle / kTwoPi
    else:
        # update relative normalized pos_x of the other agent
        obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = float(
            loc_x_arr[kEnvId, other_agent_id] - loc_x_arr[kEnvId, kThisAgentId]
        ) / (math.sqrt(2.0) * kStageSize)
        # update relative normalized pos_y of the other agent
        obs_arr[kEnvId, kThisAgentId, 5 * (kNumAgents - 1) + index] = float(
            loc_y_arr[kEnvId, other_agent_id] - loc_y_arr[kEnvId, kThisAgentId]
        ) / (math.sqrt(2.0) * kStageSize)
    return obs_arr

# Device helper function to generate observation
@numba_driver.jit((float32[:, ::1],  # loc_x_arr
                   float32[:, ::1],  # loc_y_arr
                   float32[:, ::1],  # speed_arr
                   float32[:, ::1],  # orientation_arr
                   float32[:, ::1],  # acceleration_arr
                   int32[::1],  # agent_types_arr
                   float32,  # kStageSize
                   float32,  # kMaxSpeed
                   int32[:, ::1],  # still_in_the_game_arr
                   boolean,  # kUseFullObservation
                   int32,  # kNumOtherAgentsObserved
                   float32[:, :, ::1], # neighbor_distances_arr
                   int32[:, :, ::1], # neighbor_ids_sorted_by_distance_arr
                   int32[:, :, ::1], # nearest_neighbor_ids
                   float32,  # kMaxSeeingAngle
                   float32,  # kMaxSeeingDistance
                   boolean,  # kUseTimeInObservation
                   boolean,  # kUsePolarCoordinate
                   float32[:, :, ::1],  # obs_arr
                   int32[::1],  # env_timestep_arr
                   int32,  # kNumAgents
                   int32,  # kEpisodeLength
                   int32,  # kEnvId
                   int32),  # kThisAgentId
                  device=True)
def CudaCustomEnvGenerateObservation(
        loc_x_arr,
        loc_y_arr,
        speed_arr,
        orientation_arr,
        acceleration_arr,
        agent_types_arr,
        kStageSize,
        kMaxSpeed,
        still_in_the_game_arr,
        kUseFullObservation,
        kNumOtherAgentsObserved,
        neighbor_distances_arr,
        neighbor_ids_sorted_by_distance_arr,
        nearest_neighbor_ids,    
        kMaxSeeingAngle,
        kMaxSeeingDistance,   
        kUseTimeInObservation,
        kUsePolarCoordinate,
        obs_arr,
        env_timestep_arr,
        kNumAgents,
        kEpisodeLength,
        kEnvId,
        kThisAgentId,
):
    # for each agent, we have 6 features that we can observe
    num_features = 6

    if kThisAgentId < kNumAgents:

        # FULL OBSERVATION
        if kUseFullObservation:
            index = 0

            # Initialize obs of other agents to 0 physics variable and to agent type and still in the game
            for other_agent_id in range(kNumAgents):
                if not other_agent_id == kThisAgentId:
                    obs_arr[kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index] = agent_types_arr[other_agent_id]
                    obs_arr[kEnvId, kThisAgentId, 1 * (kNumAgents - 1) + index] = still_in_the_game_arr[kEnvId, other_agent_id]
                    obs_arr[kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 5 * (kNumAgents - 1) + index] = 0.0
                    index += 1

            if kUseTimeInObservation:
                obs_arr[kEnvId, kThisAgentId, num_features * (kNumAgents - 1)] = 0.0
                
            # Update obs for agents still in the game
            if still_in_the_game_arr[kEnvId, kThisAgentId]:
                index = 0
                # Update obs of other agents
                for other_agent_id in range(kNumAgents):
                    if not other_agent_id == kThisAgentId:
                        obs_arr = UpdateObservation(
                            obs_arr,
                            kEnvId, 
                            kThisAgentId, 
                            kNumAgents, 
                            index, 
                            other_agent_id, 
                            loc_x_arr,
                            loc_y_arr,
                            kStageSize,
                            speed_arr,
                            kMaxSpeed,
                            orientation_arr,
                            kUsePolarCoordinate,
                        )
                        index += 1
                        
                if kUseTimeInObservation:
                    # add the time remaining in the episode as the last feature
                    obs_arr[kEnvId, kThisAgentId, num_features * (kNumAgents - 1)] = (
                            float(env_timestep_arr[kEnvId]) / kEpisodeLength
                    )

        # BASED ON NUMBER 
        elif kNumOtherAgentsObserved<kNumAgents:
            # Initialize obs to all zeros
            for idx in range(kNumOtherAgentsObserved):
                obs_arr[kEnvId, kThisAgentId, 0 * kNumOtherAgentsObserved + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 1 * kNumOtherAgentsObserved + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 2 * kNumOtherAgentsObserved + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 3 * kNumOtherAgentsObserved + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 4 * kNumOtherAgentsObserved + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 5 * kNumOtherAgentsObserved + idx] = 0.0
            
            if kUseTimeInObservation:
                obs_arr[kEnvId, kThisAgentId, num_features * kNumOtherAgentsObserved] = 0.0

            # Update obs for agents still in the game
            if still_in_the_game_arr[kEnvId, kThisAgentId]:

                # Initialize neighbor_ids_sorted_by_distance_arr
                # other agents that are still in the same
                num_valid_other_agents = 0
                for other_agent_id in range(kNumAgents):
                    if not other_agent_id == kThisAgentId and still_in_the_game_arr[kEnvId, other_agent_id]:
                        neighbor_ids_sorted_by_distance_arr[
                            kEnvId, kThisAgentId, num_valid_other_agents
                        ] = other_agent_id
                        num_valid_other_agents += 1

                # First, find distance to all the valid agents
                for idx in range(num_valid_other_agents):
                    neighbor_distances_arr[kEnvId, kThisAgentId, idx] = ComputeDistance(
                        loc_x_arr,
                        loc_y_arr,
                        kThisAgentId,
                        neighbor_ids_sorted_by_distance_arr[kEnvId, kThisAgentId, idx],
                        kEnvId,
                    )

                # Find the nearest neighbor agent indices
                for i in range(min(num_valid_other_agents, kNumOtherAgentsObserved)):
                    for j in range(i + 1, num_valid_other_agents):

                        if neighbor_distances_arr[kEnvId, kThisAgentId, j] < neighbor_distances_arr[kEnvId, kThisAgentId, i] :
                            tmp1 = neighbor_distances_arr[kEnvId, kThisAgentId, i]
                            neighbor_distances_arr[kEnvId, kThisAgentId, i] = neighbor_distances_arr[kEnvId, kThisAgentId, j]
                            neighbor_distances_arr[kEnvId, kThisAgentId, j] = tmp1

                            tmp2 = neighbor_ids_sorted_by_distance_arr[kEnvId, kThisAgentId, i]
                            neighbor_ids_sorted_by_distance_arr[kEnvId, kThisAgentId, i] = neighbor_ids_sorted_by_distance_arr[kEnvId, kThisAgentId, j]
                            neighbor_ids_sorted_by_distance_arr[kEnvId, kThisAgentId, j] = tmp2


                # Save nearest neighbor ids
                for idx in range(min(num_valid_other_agents, kNumOtherAgentsObserved)):
                    nearest_neighbor_ids[
                        kEnvId, kThisAgentId, idx
                    ] = neighbor_ids_sorted_by_distance_arr[kEnvId, kThisAgentId, idx]

                # Update observation
                for idx in range(min(num_valid_other_agents, kNumOtherAgentsObserved)):
                    kOtherAgentId = nearest_neighbor_ids[kEnvId, kThisAgentId, idx]
                    obs_arr[kEnvId, kThisAgentId, 0 * kNumOtherAgentsObserved + idx] = agent_types_arr[kOtherAgentId]
                    obs_arr[kEnvId, kThisAgentId, 1 * kNumOtherAgentsObserved + idx] = still_in_the_game_arr[kEnvId, kOtherAgentId]
                    obs_arr = UpdateObservation(
                        obs_arr,
                        kEnvId, 
                        kThisAgentId, 
                        kNumAgents, 
                        index, 
                        other_agent_id, 
                        loc_x_arr,
                        loc_y_arr,
                        kStageSize,
                        speed_arr,
                        kMaxSpeed,
                        orientation_arr,
                        kUsePolarCoordinate,
                    )
                if kUseTimeInObservation:
                    obs_arr[kEnvId, kThisAgentId, num_features * kNumOtherAgentsObserved] = (
                        float(env_timestep_arr[kEnvId]) / kEpisodeLength
                    )
                  
        # BASED ON DISTANCE 
        else:
            index = 0

            # Initialize obs of other agents to 0 physics variable and to agent type and still in the game
            for other_agent_id in range(kNumAgents):
                if not other_agent_id == kThisAgentId:
                    obs_arr[kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index] = agent_types_arr[other_agent_id]
                    obs_arr[kEnvId, kThisAgentId, 1 * (kNumAgents - 1) + index] = still_in_the_game_arr[kEnvId, other_agent_id]
                    obs_arr[kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 5 * (kNumAgents - 1) + index] = 0.0
                    index += 1
            if kUseTimeInObservation:
                obs_arr[kEnvId, kThisAgentId, num_features * (kNumAgents - 1)] = 0.0
                
            # Update obs for agents still in the game
            if still_in_the_game_arr[kEnvId, kThisAgentId]:
                index = 0
                # Update obs of other agents
                for other_agent_id in range(kNumAgents):
                    if not other_agent_id == kThisAgentId:
                        ditance = ComputeDistance(loc_x_arr, loc_y_arr, kThisAgentId, other_agent_id, kEnvId)
                        angle = ComputeAngle(loc_x_arr, loc_y_arr, orientation_arr, kThisAgentId, other_agent_id, kEnvId)
                        if ditance<kMaxSeeingDistance and angle<kMaxSeeingAngle:
                            obs_arr = UpdateObservation(
                                obs_arr,
                                kEnvId, 
                                kThisAgentId, 
                                kNumAgents, 
                                index, 
                                other_agent_id, 
                                loc_x_arr,
                                loc_y_arr,
                                kStageSize,
                                speed_arr,
                                kMaxSpeed,
                                orientation_arr,
                                kUsePolarCoordinate,
                            )
                            index += 1

                # add the time remaining in the episode as the last feature
                if kUseTimeInObservation:
                    obs_arr[kEnvId, kThisAgentId, num_features * (kNumAgents - 1)] = (
                            float(env_timestep_arr[kEnvId]) / kEpisodeLength
                    )

# Device helper function to compute rewards
@numba_driver.jit((float32[:, ::1],  # rewards_arr
                   float32[:, ::1],  # loc_x_arr
                   float32[:, ::1],  # loc_y_arr
                   float32,  # kStageSize
                   float32[:, ::1],  # edge_hit_reward_penalty_arr
                   int32[::1],  # num_preys_arr
                   int32[::1],  # num_predators_arr
                   int32[::1],  # agent_types_arr
                   float32[:, ::1],  # energy_cost_penalty_arr
                   float32,  # kStarvingPenaltyForPredator
                   float32,  # kEatingRewardForPredator
                   float32,  # kSurvivingRewardForPrey
                   float32,  # kDeathPenaltyForPrey
                   float32,  # kEndOfGameReward
                   float32,  # kEndOfGamePenalty
                   int32[:, ::1],  # still_in_the_game_arr
                   int32[::1],  # done_arr
                   int32[::1],  # env_timestep_arr
                   float32,  # kEatingDistance
                   int32,  # kNumAgents
                   int32,  # kEpisodeLength
                   int32,  # kEnvId
                   int32),  # kThisAgentId
                  device=True)
def CudaCustomEnvComputeReward(
        rewards_arr,
        loc_x_arr,
        loc_y_arr,
        kStageSize,
        edge_hit_reward_penalty_arr,
        num_preys_arr,
        num_predators_arr,
        agent_types_arr,
        energy_cost_penalty_arr,
        kStarvingPenaltyForPredator,
        kEatingRewardForPredator,
        kSurvivingRewardForPrey,
        kDeathPenaltyForPrey,
        kEndOfGameReward,
        kEndOfGamePenalty,
        still_in_the_game_arr,
        done_arr,
        env_timestep_arr,
        kEatingDistance,
        kNumAgents,
        kEpisodeLength,
        kEnvId,
        kThisAgentId,
):
    if kThisAgentId < kNumAgents:
        # Initialize rewards
        rewards_arr[kEnvId, kThisAgentId] = 0
        # Ensure that all the agents rewards are initialized before we proceed
        numba_driver.syncthreads()

        if still_in_the_game_arr[kEnvId, kThisAgentId]:
            min_dist = kStageSize * math.sqrt(2.0)
            is_prey = not agent_types_arr[kThisAgentId] # 0 for prey, 1 for predator

            if is_prey:
                rewards_arr[kEnvId, kThisAgentId] = kSurvivingRewardForPrey
                for other_agent_id in range(kNumAgents):
                    
                    #We compare distance with predators
                    is_predator = agent_types_arr[other_agent_id] == 1
                    if is_predator:
                        dist = ComputeDistance(
                            loc_x_arr,
                            loc_y_arr,
                            kThisAgentId,
                            other_agent_id,
                            kEnvId,
                        )
                        if dist < min_dist:
                            min_dist = dist
                            nearest_predator_id = other_agent_id
            else:
                rewards_arr[kEnvId, kThisAgentId] = kStarvingPenaltyForPredator

            if min_dist < kEatingDistance:
                # The prey is eaten
                still_in_the_game_arr[kEnvId, kThisAgentId] = 0
                num_preys_arr[kEnvId] -= 1
                
                # The reward changes
                #for agent_id in range(kNumAgents):
                #if still_in_the_game_arr[kEnvId, agent_id]:
                #is_predator = agent_types_arr[agent_id] == 1
                #if is_predator:
                rewards_arr[kEnvId, nearest_predator_id] = kEatingRewardForPredator
                #else:
                rewards_arr[kEnvId, kThisAgentId] = kDeathPenaltyForPrey

            # Add end of the game reward if useful
            if env_timestep_arr[kEnvId] == kEpisodeLength:
                rewards_arr[kEnvId, kThisAgentId] = kEndOfGameReward
            
            # Add the edge hit penalty
            rewards_arr[kEnvId, kThisAgentId] += edge_hit_reward_penalty_arr[
                kEnvId, kThisAgentId
            ]
            # Add the energy efficiency penalty
            rewards_arr[kEnvId, kThisAgentId] += energy_cost_penalty_arr[kEnvId, kThisAgentId]/2
            
#            if num_preys_arr[kEnvId] < kNumAgents-num_predators_arr[kEnvId]:
#                is_predator = agent_types_arr[kThisAgentId] == 1
#                if is_predator:
#                    rewards_arr[kEnvId, kThisAgentId] += kEndOfGameReward
#                else:
#                    rewards_arr[kEnvId, kThisAgentId] += kEndOfGamePenalty
#            else:
#                if is_prey:
#                    rewards_arr[kEnvId, kThisAgentId] += kEndOfGameReward
#                else:
#                    rewards_arr[kEnvId, kThisAgentId] += kEndOfGamePenalty
#                    

            
    numba_driver.syncthreads()
    
    if kThisAgentId == 0:
        if env_timestep_arr[kEnvId] == kEpisodeLength:
            done_arr[kEnvId] = 1

@numba_driver.jit((float32[:, ::1],  # loc_x_arr
                   float32[:, ::1],  # loc_y_arr
                   float32[:, ::1],  # speed_arr
                   float32[:, ::1],  # orientation_arr
                   float32[:, ::1],  # acceleration_arr
                   int32[::1],  # agent_types_arr
                   float32,  # kStageSize
                   float32[::1],  # acceleration_actions_arr
                   float32[::1],  # turn_actions_arr
                   float32,  # kMinSpeed
                   float32,  # kMaxSpeed
                   float32,  # kMaxAcceleration
                   float32,  # kMinAcceleration
                   float32,  # kMaxTurn
                   float32,  # kMinTurn
                   float32,  # kEatingDistance
                   float32,  # kDragingForceCoefficient
                   int32[:, ::1],  # still_in_the_game_arr
                   float32[:, :, ::1],  # obs_arr
                   boolean,  # kUseFullObservation
                   int32,  # kNumOtherAgentsObserved
                   float32[:, :, ::1], # neighbor_distances_arr
                   int32[:, :, ::1], # neighbor_ids_sorted_by_distance_arr
                   int32[:, :, ::1], # nearest_neighbor_ids
                   float32,  # kMaxSeeingAngle
                   float32,  # kMaxSeeingDistance
                   boolean,  # kUseTimeInObservation
                   boolean,  # kUsePolarCoordinate
                   int32[:, :, ::1],  # action_indices_arr
                   float32[:, ::1],  # rewards_arr
                   int32[::1],  # num_preys_arr
                   int32[::1],  # num_predators_arr
                   float32,  # kEdgeHitPenalty
                   float32,  # kStarvingPenaltyForPredator
                   float32,  # kEatingRewardForPredator
                   float32,  # kSurvivingRewardForPrey
                   float32,  # kDeathPenaltyForPrey
                   float32,  # kEndOfGameReward
                   float32,  # kEndOfGamePenalty
                   boolean,  # kUseEnergyCost
                   float32[:, ::1],  # edge_hit_reward_penalty_arr
                   float32[:, ::1],  # energy_cost_penalty_arr
                   int32[::1],  # done_arr
                   int32[::1],  # env_timestep_arr
                   int32,  # kNumAgents
                   int32))  # kEpisodeLength
def NumbaCustomEnvStep(
        loc_x_arr,
        loc_y_arr,
        speed_arr,
        orientation_arr,
        acceleration_arr,
        agent_types_arr,
        kStageSize,
        acceleration_actions_arr,
        turn_actions_arr,
        kMinSpeed,
        kMaxSpeed,
        kMaxAcceleration,
        kMinAcceleration,
        kMaxTurn,
        kMinTurn,
        kEatingDistance,
        kDragingForceCoefficient,
        still_in_the_game_arr,
        obs_arr,
        kUseFullObservation,
        kNumOtherAgentsObserved,
        neighbor_distances_arr,
        neighbor_ids_sorted_by_distance_arr,
        nearest_neighbor_ids,
        kMaxSeeingAngle,
        kMaxSeeingDistance,
        kUseTimeInObservation,
        kUsePolarCoordinate,
        action_indices_arr,
        rewards_arr,
        num_preys_arr,
        num_predators_arr,
        kEdgeHitPenalty,
        kStarvingPenaltyForPredator,
        kEatingRewardForPredator,
        kSurvivingRewardForPrey,
        kDeathPenaltyForPrey,
        kEndOfGameReward,
        kEndOfGamePenalty,
        kUseEnergyCost,
        edge_hit_reward_penalty_arr,
        energy_cost_penalty_arr,
        done_arr,
        env_timestep_arr,
        kNumAgents,
        kEpisodeLength,
):
    # Every block is an environment
    # Every thread is an agent
    kEnvId = numba_driver.blockIdx.x
    kThisAgentId = numba_driver.threadIdx.x

    # this is because the agent will take two actions : acceleration and turn
    kNumActions = 2

    # At every call, each agent steps count is incremented
    if kThisAgentId == 0:
        env_timestep_arr[kEnvId] += 1

    numba_driver.syncthreads()

    # We make sure that the timestep is valid
    assert env_timestep_arr[kEnvId] > 0 and env_timestep_arr[kEnvId] <= kEpisodeLength

    if kThisAgentId < kNumAgents:
        # get the actions for this agent
        self_force_amplitude = acceleration_actions_arr[action_indices_arr[kEnvId, kThisAgentId, 0]]
        self_force_orientation = orientation_arr[kEnvId, kThisAgentId] + turn_actions_arr[action_indices_arr[kEnvId, kThisAgentId, 1]]
        # set the energy cost penalty
        if kUseEnergyCost:
            energy_cost_penalty_arr[kEnvId, kThisAgentId] = - (self_force_amplitude/kMaxAcceleration + abs(self_force_orientation)/kMaxTurn)/100

        # draging_force
        dragging_force_amplitude = speed_arr[kEnvId, kThisAgentId] * kDragingForceCoefficient
        dragging_force_orientation = orientation_arr[kEnvId, kThisAgentId]-math.pi

        # Compute the acceleration and turn using projection
        acceleration_x = self_force_amplitude * math.cos(self_force_orientation) + dragging_force_amplitude * math.cos(
            dragging_force_orientation
        )
        
        acceleration_y = self_force_amplitude * math.sin(self_force_orientation) + dragging_force_amplitude * math.sin(
            dragging_force_orientation
        )
        # Compute the amplitude and turn in polar coordinate
        acceleration_amplitude_arr = math.sqrt(acceleration_x ** 2 + acceleration_y ** 2)
        acceleration_orientation_arr = math.atan2(acceleration_y, acceleration_x)
        
        # Compute the acceleration and turn using projection
        speed_x = speed_arr[kEnvId, kThisAgentId] * math.cos(orientation_arr[kEnvId, kThisAgentId]) + acceleration_x
        speed_y = speed_arr[kEnvId, kThisAgentId] * math.sin(orientation_arr[kEnvId, kThisAgentId]) + acceleration_y

        # Update the agent's acceleration and directions
        acceleration_arr[kEnvId, kThisAgentId] = acceleration_amplitude_arr * still_in_the_game_arr[kEnvId, kThisAgentId]
        orientation_arr[kEnvId, kThisAgentId] = math.atan2(speed_y, speed_x) 
        speed_arr[kEnvId, kThisAgentId] = math.sqrt(speed_x ** 2 + speed_y ** 2) * still_in_the_game_arr[kEnvId, kThisAgentId]
        
        # Speed clipping
        # if speed_arr[kEnvId, kThisAgentId] > kMaxSpeed :
        #     speed_arr[kEnvId, kThisAgentId] = kMaxSpeed * still_in_the_game_arr[kEnvId, kThisAgentId]

        # Reset acceleration to 0 when speed becomes 0 or
        if speed_arr[kEnvId, kThisAgentId] <= kMinSpeed or speed_arr[kEnvId, kThisAgentId] >= kMaxSpeed:
            acceleration_arr[kEnvId, kThisAgentId] = 0.0
            

        # Update the agent's location
        loc_x_arr[kEnvId, kThisAgentId] += speed_arr[kEnvId, kThisAgentId] * math.cos(
            orientation_arr[kEnvId, kThisAgentId]
        )
        loc_y_arr[kEnvId, kThisAgentId] += speed_arr[kEnvId, kThisAgentId] * math.sin(
            orientation_arr[kEnvId, kThisAgentId]
        )

        # BUMP INTO OTHER AGENTS
        
        
        # EDGE CROSSING

        # Check if the agent has crossed the edge
        has_crossed_edge = (
                loc_x_arr[kEnvId, kThisAgentId] < 0
                or loc_x_arr[kEnvId, kThisAgentId] > kStageSize
                or loc_y_arr[kEnvId, kThisAgentId] < 0
                or loc_y_arr[kEnvId, kThisAgentId] > kStageSize
        )

        # Clip x and y if agent has crossed edge
        # Here we should code a bounce back effect
        if has_crossed_edge:
            if loc_x_arr[kEnvId, kThisAgentId] < 0:
                loc_x_arr[kEnvId, kThisAgentId] = 0.0
            elif loc_x_arr[kEnvId, kThisAgentId] > kStageSize:
                loc_x_arr[kEnvId, kThisAgentId] = kStageSize

            if loc_y_arr[kEnvId, kThisAgentId] < 0:
                loc_y_arr[kEnvId, kThisAgentId] = 0.0
            elif loc_y_arr[kEnvId, kThisAgentId] > kStageSize:
                loc_y_arr[kEnvId, kThisAgentId] = kStageSize

            edge_hit_reward_penalty_arr[kEnvId, kThisAgentId] = kEdgeHitPenalty
        else:
            edge_hit_reward_penalty_arr[kEnvId, kThisAgentId] = 0.0

    numba_driver.syncthreads()

    # --------------------------------
    # Generate observation           -
    # --------------------------------

    CudaCustomEnvGenerateObservation(
        loc_x_arr,
        loc_y_arr,
        speed_arr,
        orientation_arr,
        acceleration_arr,
        agent_types_arr,
        kStageSize,
        kMaxSpeed,
        still_in_the_game_arr,
        kUseFullObservation,
        kNumOtherAgentsObserved,
        neighbor_distances_arr,
        neighbor_ids_sorted_by_distance_arr,
        nearest_neighbor_ids,    
        kMaxSeeingAngle,
        kMaxSeeingDistance,     
        kUseTimeInObservation,
        kUsePolarCoordinate,
        obs_arr,
        env_timestep_arr,
        kNumAgents,
        kEpisodeLength,
        kEnvId,
        kThisAgentId,
    )

    # --------------------------------
    # Compute reward                 -
    # --------------------------------
    CudaCustomEnvComputeReward(
        rewards_arr,
        loc_x_arr,
        loc_y_arr,
        kStageSize,
        edge_hit_reward_penalty_arr,
        num_preys_arr,
        num_predators_arr,
        agent_types_arr,
        energy_cost_penalty_arr,
        kStarvingPenaltyForPredator,
        kEatingRewardForPredator,
        kSurvivingRewardForPrey,
        kDeathPenaltyForPrey,
        kEndOfGameReward,
        kEndOfGamePenalty,
        still_in_the_game_arr,
        done_arr,
        env_timestep_arr,
        kEatingDistance,
        kNumAgents,
        kEpisodeLength,
        kEnvId,
        kThisAgentId,
    )
