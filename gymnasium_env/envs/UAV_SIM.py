from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from miscellaneous.Build_UAV_Model import Build_UAV_Model

# class Actions(Enum):
#     right = 0
#     up = 1
#     left = 2
#     down = 3


class UAVEnv(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.dt = 0.01
        self.x = np.zeros(13)

        self.A, self.B, buffer_lon, buffer_lat, buffer_col, buffer_ped, constant_vals = Build_UAV_Model()


        self.delay_buffers = {
            "lat": buffer_lat,
            "lon": buffer_lon,
            "col": buffer_col,
            "ped": buffer_ped,
        }

        # a 13 dimensional vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        # assumption is normalized input (naturally 4 dimensional)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))

        # """
        # The following dictionary maps abstract actions from `self.action_space` to
        # the direction we will walk in if that action is taken.
        # i.e. 0 corresponds to "right", 1 to "up" etc.
        # """
        # self._action_to_direction = {
        #     Actions.right.value: np.array([1, 0]),
        #     Actions.up.value: np.array([0, 1]),
        #     Actions.left.value: np.array([-1, 0]),
        #     Actions.down.value: np.array([0, -1]),
        # }

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode
        #
        # """
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        # """
        # self.window = None
        # self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x[:] = 0.0

        for buf in self.delay_buffers.values():
            buf_len = len(buf)
            buf.clear()
            buf.extend([0.0] * buf_len)

        return self.x.copy(), {}

    def step(self, action):
        # # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if terminated else 0  # Binary sparse rewards
        # observation = self._get_obs()
        # info = self._get_info()
        #
        # if self.render_mode == "human":
        #     self._render_frame()

        self.delay_buffers["lon"].append(action[1])
        self.delay_buffers["lat"].append(action[0])
        self.delay_buffers["col"].append(action[2])
        self.delay_buffers["ped"].append(action[3])

        delayed_input = np.array([self.delay_buffers["lon"][0], self.delay_buffers["lat"][0], self.delay_buffers["col"][0], self.delay_buffers["ped"][0]])

        dx = self.A @ self.x + self.B @ delayed_input
        self.x = self.x + dx * self.dt

        dp = self.x[3]
        dq = self.x[4]
        dr = self.x[5]

        dtheta = self.x[6]
        dphi = self.x[7]
        dpsi = self.x[8]

        reward = -(dphi ** 2 + dtheta ** 2 + dp ** 2 + dq ** 2 + dr ** 2)

        terminated = False
        truncated = False

        return self.x.copy(), reward, terminated, truncated, {}

    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()

    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()
    #
    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #     pix_square_size = (
    #         self.window_size / self.size
    #     )  # The size of a single grid square in pixels
    #
    #     # First we draw the target
    #     pygame.draw.rect(
    #         canvas,
    #         (255, 0, 0),
    #         pygame.Rect(
    #             pix_square_size * self._target_location,
    #             (pix_square_size, pix_square_size),
    #         ),
    #     )
    #     # Now we draw the agent
    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #     )
    #
    #     # Finally, add some gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=3,
    #         )
    #
    #     if self.render_mode == "human":
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()
    #
    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to
    #         # keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )
    #
    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
