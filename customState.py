from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
import random
import numpy as np
import math

class CustomState(StateSetter):

    SPAWN_BLUE_POS = [[-1048, -1560, 17], [1048, -1560, 17],
                      [-256, -1840, 17], [256, -1840, 17], [0, -1608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[1048, 1560, 17], [-1048, 1560, 17],
                        [256, 1840, 17], [-256, 1840, 17], [0, 1608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        random.shuffle(spawn_inds)
        sign = self.random_sign()
        state_wrapper.ball.position[2] = np.random.rand()*500+1300
        state_wrapper.ball.position[0], state_wrapper.ball.position[1], d = self.generate_point(0, 0, 2000)
        state_wrapper.ball.linear_velocity[2] = 800
        state_wrapper.ball.linear_velocity[0], state_wrapper.ball.linear_velocity[1], d = self.generate_point(200, 100, 800)

        blue_count = 0
        orange_count = 0
        x, y, d = self.generate_point(state_wrapper.ball.position[0],state_wrapper.ball.position[1],500)

        pos1 = [x*sign, -abs(y), 17]
        pos2 = self.other_point(pos1,state_wrapper.ball.position)
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = self.angle_between([pos1[0], pos1[1]],[state_wrapper.ball.position[0], state_wrapper.ball.position[1]])
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = pos1
                #yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                yaw=yaw
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = pos2
                #yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                yaw=yaw-np.pi
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = np.random.uniform(0.2, 1)

    def other_point(self, point, midpoint):
        x1, y1, z1 = point
        x2, y2, z2 = midpoint
        x_other = 2 * x2 - x1
        y_other = 2 * y2 - y1
        return [x_other, y_other, 17]

    def random_sign(self):
        if random.randint(0, 1) == 0:
            return 1
        else:
            return -1

    def generate_point(self,x, y, d):
        # Generate random values for the x and y coordinates of the new point
        x2 = random.uniform(x - d, x + d)
        y2 = random.uniform(y - d, y + d)

        # Calculate the distance between the new point and the original point
        distance = math.sqrt((x2 - x)**2 + (y2 - y)**2)

        # If the distance is greater than or equal to d, return the new point
        if distance <= d:
            return x2, y2, distance
        # Otherwise, recursively generate a new point until the distance is at least d
        else:
            return self.generate_point(x, y, d)

    def angle_between(self, a, b):
        dot_product = a[0] * b[0] + a[1] * b[1]

        # Calculate the magnitudes of the two vectors
        magnitude_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
        magnitude_b = math.sqrt(b[0] ** 2 + b[1] ** 2)

        # Calculate the cosine of the angle between the two lines
        cosine = dot_product / (magnitude_a * magnitude_b)

        # Calculate the angle in radians using the inverse cosine
        angle_radians = math.acos(cosine)

        return angle_radians
