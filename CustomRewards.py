import time

import numpy as np

from rlgym.utils import RewardFunction, math
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y, CAR_MAX_SPEED
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

class VelocityPlayerReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        for other in state.players:
            if other.car_id != player.car_id:
                return max(min(np.linalg.norm(player.car_data.linear_velocity) - 1400, 1.0), -1.0)

class EnemyTouchBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        for other in state.players:
            if other.team_num != player.team_num and other.ball_touched:
                return -(state.ball.position[2]) / BALL_RADIUS
        return 0

class BoostUseReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return previous_action[-2]

class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        dot_product = np.multiply(vel, pos_diff)/np.linalg.norm(pos_diff)
        return max(min(np.linalg.norm(dot_product)/CAR_MAX_SPEED, 1.0), -1.0)

class RewardIfBehindBall(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] \
               or player.team_num == ORANGE_TEAM and player.car_data.position[1] > state.ball.position[1]

class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1., offense=1.):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(ball - pos, attacc - pos)
        return defensive_reward + offensive_reward

class ClosestToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        for other in state.players:
            if other.team_num != player.team_num and np.linalg.norm(state.ball.position-other.car_data.position) < np.linalg.norm(state.ball.position-player.car_data.position):
                return -1
        return 1

class DriveForwardReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if previous_action[0]:
            return 1
        return 0

class JumpHighToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[-1] > 250.0:
            if player.car_data.position[-1] > state.ball.position[-1] + 150:
                return state.ball.position[-1]-player.car_data.position[-1] / min(player.car_data.position[-1], state.ball.position[-1])
            for other in state.players:
                if other.car_id != player.car_id:
                    return (player.car_data.position[-1] - other.car_data.position[-1])/min(player.car_data.position[-1], other.car_data.position[-1])
        return 0

class JumpToBallVelocityReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position

        if not player.on_ground:
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))
        else:
            return 0

class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92):
        self.min_height = min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if abs(player.car_data.position[0]) < 4001 and abs(player.car_data.position[1]) <5010:
            if player.ball_touched and not player.on_ground and state.ball.position[2] >= 1.5*self.min_height:
                return state.ball.position[2] / self.min_height
            elif player.ball_touched:
                return np.linalg.norm(state.ball.linear_velocity)/CAR_MAX_SPEED
        return 0

class DribbleReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched \
                and 11 / 10 * np.linalg.norm(state.ball.linear_velocity) >= \
                np.linalg.norm(player.car_data.linear_velocity) >= \
                9 / 10 * np.linalg.norm(state.ball.linear_velocity) \
                and player.car_data.position[2]<state.ball.position[2]:
            return np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED
        return 0


class FirstTouchReward(RewardFunction):
    def __init__(self, condition: TimeoutCondition):
        super().__init__()
        self.condition = condition

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            return 2000/(self.condition.steps+0.1)
        return 0


class AirTimeReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.car_data.position[2]>200 and player.car_data.position[2] < state.ball.position[2]:
            return player.car_data.position[2]/100
        return 0
