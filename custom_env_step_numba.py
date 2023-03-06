import math

import numba.cuda as numba_driver
from numba import float32, int32, boolean

kTwoPi = 6.283185308
kEpsilon = 1.0e-10


# Device helper function to compute distances between two agents
@numba_driver.jit((float32[:, ::1],
                   float32[:, ::1],
                   int32,
                   int32,
                   int32,
                   int32),
                  device=True,
                  inline=True)
def ComputeDistance(
        loc_x_arr, loc_y_arr, kThisAgentId1, kThisAgentId2, kEnvId, kNumAgents
):
    return math.sqrt(
        ((loc_x_arr[kEnvId, kThisAgentId1] - loc_x_arr[kEnvId, kThisAgentId2]) ** 2)
        + ((loc_y_arr[kEnvId, kThisAgentId1] - loc_y_arr[kEnvId, kThisAgentId2]) ** 2)
    )


# Device helper function to generate observation
@numba_driver.jit((float32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   int32[::1],
                   float32,
                   float32,
                   int32[:, ::1],
                   boolean,
                   float32[:, :, ::1],
                   int32[::1],
                   int32,
                   int32,
                   int32,
                   int32),
                  device=True)
def CudaCustomEnvGenerateObservation(
        loc_x_arr,
        loc_y_arr,
        speed_arr,
        direction_arr,
        acceleration_arr,
        agent_types_arr,
        kGridLength,
        kMaxSpeed,
        still_in_the_game_arr,
        kUseFullObservation,
        obs_arr,  
        env_timestep_arr,
        kNumAgents,
        kEpisodeLength,
        kEnvId,
        kThisAgentId,
):
    num_features = 7

    if kThisAgentId < kNumAgents:
        if kUseFullObservation:
            # Initialize obs
            index = 0

            for other_agent_id in range(kNumAgents):
                if not other_agent_id == kThisAgentId:
                    obs_arr[kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index] = 0.0
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
                for other_agent_id in range(kNumAgents):
                    if not other_agent_id == kThisAgentId:
                        obs_arr[
                            kEnvId, kThisAgentId, 0 * (kNumAgents - 1) + index
                        ] = float(
                            loc_x_arr[kEnvId, other_agent_id]
                            - loc_x_arr[kEnvId, kThisAgentId]
                        ) / (
                            math.sqrt(2.0) * kGridLength
                        )
                        obs_arr[
                            kEnvId, kThisAgentId, 1 * (kNumAgents - 1) + index
                        ] = float(
                            loc_y_arr[kEnvId, other_agent_id]
                            - loc_y_arr[kEnvId, kThisAgentId]
                        ) / (
                            math.sqrt(2.0) * kGridLength
                        )
                        obs_arr[
                            kEnvId, kThisAgentId, 2 * (kNumAgents - 1) + index
                        ] = float(
                            speed_arr[kEnvId, other_agent_id]
                            - speed_arr[kEnvId, kThisAgentId]
                        ) / (
                            kMaxSpeed + kEpsilon
                        )
                        obs_arr[
                            kEnvId, kThisAgentId, 3 * (kNumAgents - 1) + index
                        ] = float(
                            acceleration_arr[kEnvId, other_agent_id]
                            - acceleration_arr[kEnvId, kThisAgentId]
                        ) / (
                            kMaxSpeed + kEpsilon
                        )
                        obs_arr[kEnvId, kThisAgentId, 4 * (kNumAgents - 1) + index] = (
                            float(
                                direction_arr[kEnvId, other_agent_id]
                                - direction_arr[kEnvId, kThisAgentId]
                            )
                            / kTwoPi
                        )
                        index += 1

                obs_arr[kEnvId, kThisAgentId, num_features * (kNumAgents - 1)] = (
                    float(env_timestep_arr[kEnvId]) / kEpisodeLength
                )
        else:
            # Initialize obs to all zeros
            for idx in range(kNumAgents):
                obs_arr[kEnvId, kThisAgentId, 0 * kNumAgents + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 1 * kNumAgents + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 2 * kNumAgents + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 3 * kNumAgents + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 4 * kNumAgents + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 5 * kNumAgents + idx] = 0.0
                obs_arr[kEnvId, kThisAgentId, 6 * kNumAgents + idx] = 0.0

            obs_arr[kEnvId, kThisAgentId, num_features * kNumAgents] = 0.0



# Device helper function to compute rewards
@numba_driver.jit((int32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   float32,
                   float32[:, ::1],
                   int32[::1],
                   int32[::1],
                   float32,
                   float32,
                   float32,
                   float32,
                   int32[:, ::1],
                   int32[::1],
                   int32[::1],
                   int32,
                   int32,
                   int32,
                   int32),
                  device=True)
def CudaCustomEnvComputeReward(
    rewards_arr,
    loc_x_arr,
    loc_y_arr,
    kGridLength,
    edge_hit_penalty_arr,
    num_preys_arr,
    agent_types_arr,
    kEatingRewardForPredator,
    kEatingPenaltyForPreys,
    kEndOfGameRewardForPrey,
    kEndOfGameRewardForPredator,
    still_in_the_game_arr,
    done_arr,
    env_timestep_arr,
    kNumAgents,
    kEpisodeLength,
    kEnvId,
    kThisAgentId,
):
    if kThisAgentId < kNumAgents:
        # Initialize rewards
        rewards_arr[kEnvId, kThisAgentId] = 0

        if still_in_the_game_arr[kEnvId, kThisAgentId]:
            # Add the edge hit penalty and the step rewards / penalties
            rewards_arr[kEnvId, kThisAgentId] += edge_hit_penalty_arr[
                kEnvId, kThisAgentId
            ]

        # Ensure that all the agents rewards are initialized before we proceed
        # The rewards are only set by the runners, so this pause is necessary
        numba_driver.syncthreads()

        min_dist = kGridLength * math.sqrt(2.0)
        is_runner = not agent_types_arr[kThisAgentId]

        if is_runner and still_in_the_game_arr[kEnvId, kThisAgentId]:
            for other_agent_id in range(kNumAgents):
                is_tagger = agent_types_arr[other_agent_id] == 1

                if is_tagger:
                    dist = ComputeDistance(
                        loc_x_arr,
                        loc_y_arr,
                        kThisAgentId,
                        other_agent_id,
                        kEnvId,
                        kNumAgents,
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_tagger_id = other_agent_id

            if min_dist < 1: #TODO: need to change that 1
                # The runner is tagged
                rewards_arr[kEnvId, kThisAgentId] += kEatingPenaltyForPreys
                rewards_arr[kEnvId, nearest_tagger_id] += kEatingRewardForPredator

                # if kRunnerExitsGameAfterTagged: # TODO : Use to be kRunnerExitsGameAfterTagged but always true
                still_in_the_game_arr[kEnvId, kThisAgentId] = 0
                num_preys_arr[kEnvId] -= 1

            # Add end of game reward for runners at the end of the episode
            if env_timestep_arr[kEnvId] == kEpisodeLength:
                rewards_arr[kEnvId, kThisAgentId] += kEndOfGameRewardForPrey

    numba_driver.syncthreads()

    if kThisAgentId == 0:
        if env_timestep_arr[kEnvId] == kEpisodeLength or num_preys_arr[kEnvId] == 0:
            done_arr[kEnvId] = 1


@numba_driver.jit((float32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   float32[:, ::1],
                   int32[::1],
                   float32,
                   float32[::1],
                   float32[::1],
                   float32,
                   float32,
                   float32,
                   float32,
                   float32,
                   int32[:, ::1],
                   boolean,
                   float32,
                   float32,
                   float32[:, :, ::1],
                   int32[:, :, ::1],
                   float32[:, ::1],
                   int32[:, ::1],
                   int32[::1],
                   int32[::1],
                   float32,
                   float32,
                   float32,
                   float32,
                   float32,
                   int32[::1],
                   int32[::1],
                   int32,
                   int32))
def NumbaCustomEnvStep(
    loc_x_arr,
    loc_y_arr,
    speed_arr,
    direction_arr,
    acceleration_arr,
    agent_types_arr,
    kGridLength,
    acceleration_actions_arr,
    turn_actions_arr,
    kMaxSpeed,
    kMaxAcceleration,
    kMinAcceleration,
    kMaxTurn,
    kMinTurn,
    still_in_the_game_arr,
    kUseFullObservation,
    kSeeingAngle,
    kSeeingDistance,
    obs_arr,
    action_indices_arr,
    edge_hit_penalty_arr,
    rewards_arr,
    num_preys_arr,
    num_predators_arr,
    kEdgeHitPenalty,
    kEatingRewardForPredator,
    kEatingPenaltyForPreys,
    kEndOfGameRewardForPrey,
    kEndOfGameRewardForPredator,
    done_arr,
    env_timestep_arr,
    kNumAgents,
    kEpisodeLength,
):
    print("Numba step function call starting")
    kEnvId = numba_driver.blockIdx.x
    kThisAgentId = numba_driver.threadIdx.x
    kNumActions = 2

    if kThisAgentId == 0:
        env_timestep_arr[kEnvId] += 1

    numba_driver.syncthreads()

    assert env_timestep_arr[kEnvId] > 0 and env_timestep_arr[kEnvId] <= kEpisodeLength

    if kThisAgentId < kNumAgents:
        delta_acceleration = acceleration_actions_arr[action_indices_arr[kEnvId, kThisAgentId, 0]]
        delta_turn = turn_actions_arr[action_indices_arr[kEnvId, kThisAgentId, 1]]

        acceleration_arr[kEnvId, kThisAgentId] += delta_acceleration
        direction_arr[kEnvId, kThisAgentId] = ((direction_arr[kEnvId, kThisAgentId] + delta_turn) % kTwoPi) * still_in_the_game_arr[kEnvId, kThisAgentId]

        if direction_arr[kEnvId, kThisAgentId] < 0:
            direction_arr[kEnvId, kThisAgentId] = (
                kTwoPi + direction_arr[kEnvId, kThisAgentId]
            )

        # Speed clipping
        speed_arr[kEnvId, kThisAgentId] = min(
            kMaxSpeed,
            max(0.0, speed_arr[kEnvId, kThisAgentId] + acceleration_arr[kEnvId, kThisAgentId],)
        ) * still_in_the_game_arr[kEnvId, kThisAgentId]


        # Reset acceleration to 0 when speed becomes 0 or
        # kMaxSpeed (multiplied by skill levels)
        if speed_arr[kEnvId, kThisAgentId] <= 0.0 or speed_arr[kEnvId, kThisAgentId] >= kMaxSpeed:
            acceleration_arr[kEnvId, kThisAgentId] = 0.0

        loc_x_arr[kEnvId, kThisAgentId] += speed_arr[kEnvId, kThisAgentId] * math.cos(
            direction_arr[kEnvId, kThisAgentId]
        )
        loc_y_arr[kEnvId, kThisAgentId] += speed_arr[kEnvId, kThisAgentId] * math.sin(
            direction_arr[kEnvId, kThisAgentId]
        )

        # Crossing the edge
        has_crossed_edge = (
            loc_x_arr[kEnvId, kThisAgentId] < 0
            or loc_x_arr[kEnvId, kThisAgentId] > kGridLength
            or loc_y_arr[kEnvId, kThisAgentId] < 0
            or loc_y_arr[kEnvId, kThisAgentId] > kGridLength
        )

        # Clip x and y if agent has crossed edge
        if has_crossed_edge:
            if loc_x_arr[kEnvId, kThisAgentId] < 0:
                loc_x_arr[kEnvId, kThisAgentId] = 0.0
            elif loc_x_arr[kEnvId, kThisAgentId] > kGridLength:
                loc_x_arr[kEnvId, kThisAgentId] = kGridLength

            if loc_y_arr[kEnvId, kThisAgentId] < 0:
                loc_y_arr[kEnvId, kThisAgentId] = 0.0
            elif loc_y_arr[kEnvId, kThisAgentId] > kGridLength:
                loc_y_arr[kEnvId, kThisAgentId] = kGridLength

            edge_hit_penalty_arr[kEnvId, kThisAgentId] = kEdgeHitPenalty
        else:
            edge_hit_penalty_arr[kEnvId, kThisAgentId] = 0.0

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
        kGridLength,
        kMaxSpeed,
        still_in_the_game_arr,
        kUseFullObservation,
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
        kGridLength,
        edge_hit_penalty_arr,
        num_preys_arr,
        agent_types_arr,
        kEatingRewardForPredator,
        kEatingPenaltyForPreys,
        kEndOfGameRewardForPrey,
        kEndOfGameRewardForPredator,
        still_in_the_game_arr,
        done_arr,
        env_timestep_arr,
        kNumAgents,
        kEpisodeLength,
        kEnvId,
        kThisAgentId,
    )