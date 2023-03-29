from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState


class DropBallCondition(TerminalCondition):
    """
    A condition that will terminate an episode after some number of steps.
    """

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:

        return current_state.ball.position[2] < 150 or current_state.last_touch !=-1
