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
        loc_x_arr, loc_y_arr, direction_arr, kThisAgentId1, kThisAgentId2, kEnvId
):
    return math.degrees(
        math.atan2(
            loc_y_arr[kEnvId, kThisAgentId1] - loc_y_arr[kEnvId, kThisAgentId2], 
            loc_x_arr[kEnvId, kThisAgentId1] - loc_x_arr[kEnvId, kThisAgentId2]
        ) - direction_arr[kEnvId, kThisAgentId1]
    )

# Device helper function to generate observation
@numba_driver.jit((float32[:, ::1],  # loc_x_arr
                   float32[:, ::1],  # loc_y_arr
                   float32[:, ::1],  # speed_arr
                   float32[:, ::1],  # direction_arr
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
        direction_arr,
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
        obs_arr,
        env_timestep_arr,
        kNumAgents,
        kEpisodeLength,
        kEnvId,
        kThisAgentId,
):
    # for each agent, we have 7 features that we can observe
    num_features = 7

    if kThisAgentId < kNumAgents:

        # FULL OBSERVATION
        if kUseFullObservation:
            index = 0

            # Initialize obs of other agents to 0 physics variable and to agent type and still in the game
            for other_agent_id in range(kNumAgents):
                if not other_agent_id == kThisAgentId:
                    obs_arr[kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index] = 0.0 # futur position of loc_x
                    obs_arr[kEnvId, kThisAgentId, 1 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 5 * (kNumAgents - 1) + index] = agent_types_arr[other_agent_id]
                    obs_arr[kEnvId, kThisAgentId, 6 * (kNumAgents - 1) + index] = still_in_the_game_arr[kEnvId, other_agent_id]
                    index += 1

            obs_arr[kEnvId, kThisAgentId, num_features * (kNumAgents - 1)] = 0.0
                
            # Update obs for agents still in the game
            if still_in_the_game_arr[kEnvId, kThisAgentId]:
                index = 0
                # Update obs of other agents
                for other_agent_id in range(kNumAgents):
                    if not other_agent_id == kThisAgentId:
                        ditance = ComputeDistance(loc_x_arr, loc_y_arr, kThisAgentId, other_agent_id, kEnvId)
                        angle = ComputeAngle(loc_x_arr, loc_y_arr, direction_arr, kThisAgentId, other_agent_id, kEnvId)
                        # if ditance<kMaxSeeingDistance and angle<kMaxSeeingAngle:
                        # update relative normalized pos_x of the other agent
                        obs_arr[kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index] = float(
                            loc_x_arr[kEnvId, other_agent_id] - loc_x_arr[kEnvId, kThisAgentId]
                        ) / (math.sqrt(2.0) * kStageSize)
                        # update relative normalized pos_y of the other agent
                        obs_arr[kEnvId, kThisAgentId, 1 * (kNumAgents - 1) + index] = float(
                            loc_y_arr[kEnvId, other_agent_id] - loc_y_arr[kEnvId, kThisAgentId]
                        ) / (math.sqrt(2.0) * kStageSize)
                        # update relative normalized speed of the other agent
                        obs_arr[kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index] = float(
                            speed_arr[kEnvId, other_agent_id] - speed_arr[kEnvId, kThisAgentId]
                        ) / (kMaxSpeed + kEpsilon)
                        # update relative normalized acceleration of the other agent
                        obs_arr[kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index] = float(
                            acceleration_arr[kEnvId, other_agent_id] - acceleration_arr[kEnvId, kThisAgentId]
                        ) / (kMaxSpeed + kEpsilon)
                        # update relative normalized direction of the other agent
                        obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = float(
                            direction_arr[kEnvId, other_agent_id] - direction_arr[kEnvId, kThisAgentId]
                        ) / kTwoPi
                        index += 1

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
                obs_arr[kEnvId, kThisAgentId, 6 * kNumOtherAgentsObserved + idx] = 0.0

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


                # Update observation
                for idx in range(min(num_valid_other_agents, kNumOtherAgentsObserved)):
                    kOtherAgentId = nearest_neighbor_ids[kEnvId, kThisAgentId, idx]

                    obs_arr[kEnvId, kThisAgentId, 0 * kNumOtherAgentsObserved + idx] = float(
                        loc_x_arr[kEnvId, kOtherAgentId] - loc_x_arr[kEnvId, kThisAgentId]
                    ) / (math.sqrt(2.0) * kStageSize)
                    obs_arr[kEnvId, kThisAgentId, 1 * kNumOtherAgentsObserved + idx] = float(
                        loc_y_arr[kEnvId, kOtherAgentId] - loc_y_arr[kEnvId, kThisAgentId]
                    ) / (math.sqrt(2.0) * kStageSize)
                    obs_arr[kEnvId, kThisAgentId, 2 * kNumOtherAgentsObserved + idx] = float(
                        speed_arr[kEnvId, kOtherAgentId] - speed_arr[kEnvId, kThisAgentId]
                    ) / (kMaxSpeed + kEpsilon)
                    obs_arr[kEnvId, kThisAgentId, 3 * kNumOtherAgentsObserved + idx] = float(
                        acceleration_arr[kEnvId, kOtherAgentId] - acceleration_arr[kEnvId, kThisAgentId]
                    ) / (kMaxSpeed + kEpsilon)
                    obs_arr[kEnvId, kThisAgentId, 4 * kNumOtherAgentsObserved + idx] = float(
                            direction_arr[kEnvId, kOtherAgentId]
                            - direction_arr[kEnvId, kThisAgentId]
                    ) / kTwoPi

                    obs_arr[kEnvId, kThisAgentId, 5 * kNumOtherAgentsObserved + idx] = agent_types_arr[kOtherAgentId]
                    obs_arr[kEnvId, kThisAgentId, 6 * kNumOtherAgentsObserved + idx] = still_in_the_game_arr[kEnvId, kOtherAgentId]

                obs_arr[kEnvId, kThisAgentId, num_features * kNumOtherAgentsObserved] = (float(env_timestep_arr[kEnvId]) / kEpisodeLength)
                  
        # BASED ON DISTANCE 
        else:
            index = 0

            # Initialize obs of other agents to 0 physics variable and to agent type and still in the game
            for other_agent_id in range(kNumAgents):
                if not other_agent_id == kThisAgentId:
                    obs_arr[kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index] = 0.0 # futur position of loc_x
                    obs_arr[kEnvId, kThisAgentId, 1 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = 0.0
                    obs_arr[kEnvId, kThisAgentId, 5 * (kNumAgents - 1) + index] = agent_types_arr[other_agent_id]
                    obs_arr[kEnvId, kThisAgentId, 6 * (kNumAgents - 1) + index] = still_in_the_game_arr[kEnvId, other_agent_id]
                    index += 1

            obs_arr[kEnvId, kThisAgentId, num_features * (kNumAgents - 1)] = 0.0
                
            # Update obs for agents still in the game
            if still_in_the_game_arr[kEnvId, kThisAgentId]:
                index = 0
                # Update obs of other agents
                for other_agent_id in range(kNumAgents):
                    if not other_agent_id == kThisAgentId:
                        ditance = ComputeDistance(loc_x_arr, loc_y_arr, kThisAgentId, other_agent_id, kEnvId)
                        angle = ComputeAngle(loc_x_arr, loc_y_arr, direction_arr, kThisAgentId, other_agent_id, kEnvId)
                        if ditance<kMaxSeeingDistance and angle<kMaxSeeingAngle:
                            # update relative normalized pos_x of the other agent
                            obs_arr[kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index] = float(
                                loc_x_arr[kEnvId, other_agent_id] - loc_x_arr[kEnvId, kThisAgentId]
                            ) / (math.sqrt(2.0) * kStageSize)
                            # update relative normalized pos_y of the other agent
                            obs_arr[kEnvId, kThisAgentId, 1 * (kNumAgents - 1) + index] = float(
                                loc_y_arr[kEnvId, other_agent_id] - loc_y_arr[kEnvId, kThisAgentId]
                            ) / (math.sqrt(2.0) * kStageSize)
                            # update relative normalized speed of the other agent
                            obs_arr[kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index] = float(
                                speed_arr[kEnvId, other_agent_id] - speed_arr[kEnvId, kThisAgentId]
                            ) / (kMaxSpeed + kEpsilon)
                            # update relative normalized acceleration of the other agent
                            obs_arr[kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index] = float(
                                acceleration_arr[kEnvId, other_agent_id] - acceleration_arr[kEnvId, kThisAgentId]
                            ) / (kMaxSpeed + kEpsilon)
                            # update relative normalized direction of the other agent
                            obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = float(
                                direction_arr[kEnvId, other_agent_id] - direction_arr[kEnvId, kThisAgentId]
                            ) / kTwoPi
                            index += 1

                # add the time remaining in the episode as the last feature
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
                for agent_id in range(kNumAgents):
                    if still_in_the_game_arr[kEnvId, agent_id]:
                        is_predator = agent_types_arr[agent_id] == 1
                        if is_predator:
                            rewards_arr[kEnvId, agent_id] = kEatingRewardForPredator
                        else:
                            rewards_arr[kEnvId, agent_id] = kDeathPenaltyForPrey

#        if env_timestep_arr[kEnvId] == kEpisodeLength:
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
        if still_in_the_game_arr[kEnvId, kThisAgentId]:
            # Add the edge hit penalty and the step rewards / penalties
            rewards_arr[kEnvId, kThisAgentId] += edge_hit_reward_penalty_arr[
                kEnvId, kThisAgentId
            ]
            
    numba_driver.syncthreads()
    
    if kThisAgentId == 0:
        if env_timestep_arr[kEnvId] == kEpisodeLength:
            done_arr[kEnvId] = 1

@numba_driver.jit((float32[:, ::1],  # loc_x_arr
                   float32[:, ::1],  # loc_y_arr
                   float32[:, ::1],  # speed_arr
                   float32[:, ::1],  # direction_arr
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
                   int32[:, ::1],  # still_in_the_game_arr
                   float32[:, :, ::1],  # obs_arr
                   boolean,  # kUseFullObservation
                   int32,  # kNumOtherAgentsObserved
                   float32[:, :, ::1], # neighbor_distances_arr
                   int32[:, :, ::1], # neighbor_ids_sorted_by_distance_arr
                   int32[:, :, ::1], # nearest_neighbor_ids
                   float32,  # kMaxSeeingAngle
                   float32,  # kMaxSeeingDistance
                   int32[:, :, ::1],  # action_indices_arr
                   float32[:, ::1],  # edge_hit_reward_penalty_arr
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
                   float32,  # kEatingDistance
                   int32[::1],  # done_arr
                   int32[::1],  # env_timestep_arr
                   int32,  # kNumAgents
                   int32))  # kEpisodeLength
def NumbaCustomEnvStep(
        loc_x_arr,
        loc_y_arr,
        speed_arr,
        direction_arr,
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
        still_in_the_game_arr,
        obs_arr,
        kUseFullObservation,
        kNumOtherAgentsObserved,
        neighbor_distances_arr,
        neighbor_ids_sorted_by_distance_arr,
        nearest_neighbor_ids,
        kMaxSeeingAngle,
        kMaxSeeingDistance,
        action_indices_arr,
        edge_hit_reward_penalty_arr,
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
        kEatingDistance,
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
        delta_acceleration = acceleration_actions_arr[action_indices_arr[kEnvId, kThisAgentId, 0]]
        delta_turn = turn_actions_arr[action_indices_arr[kEnvId, kThisAgentId, 1]]

        # Update the agent's acceleration and directions
        acceleration_arr[kEnvId, kThisAgentId] += delta_acceleration
        direction_arr[kEnvId, kThisAgentId] = ((direction_arr[kEnvId, kThisAgentId] + delta_turn) % kTwoPi) * still_in_the_game_arr[
            kEnvId, kThisAgentId]

        # Direction is kept in [0, 2pi]
        if direction_arr[kEnvId, kThisAgentId] < 0:
            direction_arr[kEnvId, kThisAgentId] = (
                    kTwoPi + direction_arr[kEnvId, kThisAgentId]
            )

        # Speed clipping
        speed_arr[kEnvId, kThisAgentId] = min(
            kMaxSpeed,
            max(kMinSpeed, speed_arr[kEnvId, kThisAgentId] + acceleration_arr[kEnvId, kThisAgentId], )
        ) * still_in_the_game_arr[kEnvId, kThisAgentId]

        # Reset acceleration to 0 when speed becomes 0 or
        # kMaxSpeed (multiplied by skill levels)
        if speed_arr[kEnvId, kThisAgentId] <= kMinSpeed or speed_arr[kEnvId, kThisAgentId] >= kMaxSpeed:
            acceleration_arr[kEnvId, kThisAgentId] = 0.0

        # Update the agent's location
        loc_x_arr[kEnvId, kThisAgentId] += speed_arr[kEnvId, kThisAgentId] * math.cos(
            direction_arr[kEnvId, kThisAgentId]
        )
        loc_y_arr[kEnvId, kThisAgentId] += speed_arr[kEnvId, kThisAgentId] * math.sin(
            direction_arr[kEnvId, kThisAgentId]
        )

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
        direction_arr,
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
