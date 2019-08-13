from abc import ABC

import scipy as sp
import numpy as np
import gym
from gym import spaces  # are we using spaces?
from gym.utils import seeding

# register(
#     id='pga-v0',
#     entry_point='pga_env.pga_env:pga',
# )


class PgaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.profit = 200

        self.min_time = 0
        self.max_time = 30

        self.min_player = 0
        self.max_player = 1

        # decide on loss function
        self.loss_function = .1;

        # change to epsilon
        self.min_bid = 0
        # change to whatever
        self.max_bid = self.profit + (self.loss_function * (self.profit - 1))
        self.duration = sp.random.poisson(lam=15, size=100)
        self.low = np.array([self.min_bid, self.min_player, self.min_time])
        self.high = np.array([self.max_bid, self.max_player, self.max_time])

        # self.viewer = None

        # decide on action space

        self.action_space = np.arange(0, 1, 0.05)  # 0-2, incremented by .05 up to .95 so 20 entries

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, curr_time):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        last_bid, player, time_of_bid = self.state
        done = bool(curr_time >= self.duration)

        if action == 0:
            reward = 0
            return np.array(self.state), reward, done, {}

        #
        #     possible_last_bid.append([b_zero * (1+self.f) ** self.k, 0])
        #     self.k+= 1

        time_difference = curr_time - time_of_bid

        if time_difference <= 1:
            reward = -1  # is this a good call?
        elif 1 < time_difference <= 2:
            reward = 2
        elif 2 < time_difference <= 3:
            reward = 4
        elif 3 < time_difference <= 4:
            reward = 6
        elif 4 < time_difference <= 5:
            reward = 8
        elif time_difference > 6:
            reward = 10

        self.last_bid *= self.action_space[action]
        player = 1
        time_of_bid = curr_time

        self.state = (self.last_bid, player, time_of_bid)

        return np.array(self.state), reward, done, {}

    def reset(self):
        # figure out why it was a distribution
        self.state = np.array([self.min_bid, self.min_player, self.min_time])
        return np.array(self.state)


    # def _height(self, xs):
    #     return np.sin(3 * xs) * .45 + .55

    # def render(self, mode='human'):
    #     screen_width = 600
    #     screen_height = 400
    #
    #     world_width = self.max_position - self.min_position
    #     scale = screen_width / world_width
    #     carwidth = 40
    #     carheight = 20
    #
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         xs = np.linspace(self.min_position, self.max_position, 100)
    #         ys = self._height(xs)
    #         xys = list(zip((xs - self.min_position) * scale, ys * scale))
    #
    #         self.track = rendering.make_polyline(xys)
    #         self.track.set_linewidth(4)
    #         self.viewer.add_geom(self.track)
    #
    #         clearance = 10
    #
    #         l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
    #         car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         car.add_attr(rendering.Transform(translation=(0, clearance)))
    #         self.cartrans = rendering.Transform()
    #         car.add_attr(self.cartrans)
    #         self.viewer.add_geom(car)
    #         frontwheel = rendering.make_circle(carheight / 2.5)
    #         frontwheel.set_color(.5, .5, .5)
    #         frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
    #         frontwheel.add_attr(self.cartrans)
    #         self.viewer.add_geom(frontwheel)
    #         backwheel = rendering.make_circle(carheight / 2.5)
    #         backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
    #         backwheel.add_attr(self.cartrans)
    #         backwheel.set_color(.5, .5, .5)
    #         self.viewer.add_geom(backwheel)
    #         flagx = (self.goal_position - self.min_position) * scale
    #         flagy1 = self._height(self.goal_position) * scale
    #         flagy2 = flagy1 + 50
    #         flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
    #         self.viewer.add_geom(flagpole)
    #         flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
    #         flag.set_color(.8, .8, 0)
    #         self.viewer.add_geom(flag)
    #
    #     pos = self.state[0]
    #     self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
    #     self.cartrans.set_rotation(math.cos(3 * pos))
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # def get_keys_to_action(self):
    #     return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys
    #
    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None
