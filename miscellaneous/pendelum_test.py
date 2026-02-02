import gymnasium as gym
import numpy as np
from sympy.physics.units import acceleration

env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0

env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5



# https://gymnasium.farama.org/environments/classic_control/pendulum/

from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize


class myimplementation(PendulumEnv):

    def step(self, u):
        """
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        """

        angle, angular_velocity = self.state


        #copied from library
        costs = angle_normalize(angle) ** 2 + 0.1 * angular_velocity**2 + 0.001 * (u**2)

        # angular_accelerationon =  ½ mlg sin theta - input torque /(⅓ ml^2 )
        import math
        angular_acceleration =  0.5 * self.m * self.l * self.g * math.sin(angle)  - u /(0.5 *self.m *self.l **2 )

        new_angular_velocity = angular_velocity + self.dt + angular_acceleration

        next_angle = angle + self.dt * new_angular_velocity

        self.state = np.array([next_angle, new_angular_velocity])



        return self.state, -costs, False, False, {}



