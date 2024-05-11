#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:11:52 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import gym
import numpy as np

from gym import spaces, logger
from gym.utils import seeding

from webots_drone import WebotsSimulation
from webots_drone.utils import compute_distance
from webots_drone.utils import check_flight_area
from webots_drone.utils import check_collision
from webots_drone.utils import check_flipped
from webots_drone.utils import check_near_object
from webots_drone.utils import min_max_norm
from webots_drone.utils import check_same_position
from webots_drone.reward import compute_vector_reward
from webots_drone.reward import compute_visual_reward

from .preprocessor import seconds2steps
from .preprocessor import info2image
from .preprocessor import normalize_pixels
from .preprocessor import info2obs_1d
from .preprocessor import normalize_vector



class DroneEnvContinuous(gym.Env):
    """Gym enviroment to control the Fire scenario in the Webots simulator."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 30
    }

    def __init__(self, time_limit_seconds=60,  # 1 min
                 max_no_action_seconds=5,  # 5 sec
                 frame_skip=125,  # 1 sec
                 goal_threshold=5.,
                 init_altitude=25.,
                 altitude_limits=[11, 75],
                 fire_pos=2,
                 fire_dim=[7., 3.5],
                 is_pixels=True):
        # Simulation controller
        logger.info('Checking Webots connection...')
        self.sim = WebotsSimulation()
        logger.info('Connected to Webots')

        # Action space, the angles and altitud
        self.action_space = spaces.Box(low=self.action_limits[0],
                                       high=self.action_limits[1],
                                       shape=(self.action_limits.shape[-1], ),
                                       dtype=np.float32)
        # Observation space
        self.is_pixels = is_pixels
        if self.is_pixels:
            # Observation space, the drone's camera image
            self.obs_shape = (3, 84, 84)
            self.obs_type = np.uint8
            self.observation_space = spaces.Box(low=0,
                                                high=255,
                                                shape=self.obs_shape,
                                                dtype=self.obs_type)
        else:
            self.obs_shape = (22, )
            self.obs_type = np.float32
            self.observation_space = spaces.Box(low=float('-inf'),
                                                high=float('inf'),
                                                shape=self.obs_shape,
                                                dtype=self.obs_type)
        # runtime vars
        self.init_runtime_vars()
        self._max_episode_steps = seconds2steps(time_limit_seconds, frame_skip,
                                                self.sim.timestep)
        self._max_no_action_steps = seconds2steps(max_no_action_seconds, frame_skip,
                                                  self.sim.timestep)
        self._frame_skip = frame_skip
        self._frame_inter = [frame_skip - 5., frame_skip + 5.]
        self._goal_threshold = goal_threshold
        self._fire_pos = fire_pos
        self._fire_dim = fire_dim
        self.flight_area = np.array(self.sim.get_flight_area(altitude_limits))
        self.init_altitude = init_altitude
        self.viewer = None
        self.cuadrants = np.array(
            [(self.flight_area[0][0], self.flight_area[1][1]),
             (self.flight_area[1][0], self.flight_area[1][1]),
             (self.flight_area[1][0], self.flight_area[0][1]),
             (self.flight_area[0][0], self.flight_area[0][1])])
        self.cuadrants /= 2.
        self.reward_limits = [-2. - (2.21 * self._frame_inter[0]), 
                              3.21 * self._frame_inter[1]]

    @property
    def action_limits(self):
        return WebotsSimulation.get_control_ranges()

    def norm_reward(self, reward):
        reward = min_max_norm(reward - 1e-8,  # avoids zero values
                              -1., 2., self.reward_limits[0],
                              self.reward_limits[1])
        return reward


    def init_runtime_vars(self):
        self._episode_steps = 0  # time limit control
        self._no_action_steps = 0  # no action control
        self._end = False  # episode end flag
        self._prev_distance = 0  # reward helper
        self.last_info = dict()
        self.last_state = None

    def seed(self, seed=None):
        """Set seed for the environment random generations."""
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. Taken from atari_env.py
        # seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed2 = (seed1 + 1) % 2**31
        self.sim.seed(seed2)
        return [seed1, seed2]

    def set_fire_position(self, position, noise_ratio=0.):
        offset = 0
        if self.np_random.random() < noise_ratio:
            offset = self.np_random.random()
        self.sim.set_fire(position + offset, *self._fire_dim,
                          dist_threshold=self._goal_threshold * 1.5)

    def set_fire_cuadrant(self, cuadrant=None, noise_ratio=0.):
        logger.info(f"Starting fire in cuadrant {cuadrant} with {noise_ratio} noise.")
        if cuadrant is None:
            cuadrant = self.np_random.integers(4)
        fire_pos = np.array(self.cuadrants[cuadrant])
        self.set_fire_position(fire_pos, noise_ratio=noise_ratio)
    
    def get_observation_2d(self, state_data, norm=False):
        state_2d = info2image(state_data, output_size=self.obs_shape[-1])
        if norm:
            state_2d = normalize_pixels(state_2d)
        return state_2d
    
    def get_observation_1d(self, state_data, norm=False):
        state_1d = info2obs_1d(state_data)
        if norm:
            xyz_ranges = list(zip(*self.flight_area))
            xyz_velocities = [4., 4., 1.]
            state_1d = normalize_vector(state_1d, xyz_ranges, xyz_velocities)
        return state_1d

    def get_state(self):
        """Process the environment to get a state."""
        state_data = self.sim.get_data()

        if self.is_pixels:
            state = self.get_observation_2d(state_data)
        else:
            state = self.get_observation_1d(state_data)

        return state, state_data

    def compute_risk_dist(self, threshold=0.):
        return self.sim.get_risk_distance(threshold)
    
    @property
    def distance_target(self):
        return self.sim.get_risk_distance(self._goal_threshold / 2.)

    def __no_action_limit(self, position):
        if len(self.last_info.keys()) == 0:
            return False
        if check_same_position(position, self.last_info['position'], thr=0.003):
            self._no_action_steps += 1
        else:
            self._no_action_steps = 0
        return self._no_action_steps >= self._max_no_action_steps

    def __is_final_state(self, info):
        discount = 0
        # no action limit
        if self.__no_action_limit(info["position"]):
            logger.info(f"[{info['timestamp']}] Final state, Same position")
            discount -= 2.
            info['final'] = 'No Action'
        # is_flipped
        elif check_flipped(info["orientation"], info["dist_sensors"]):
            logger.info(f"[{info['timestamp']}] Final state, Flipped")
            discount -= 2.
            info['final'] = 'Flipped'

        return discount

    def __compute_penalization(self, info):
        near_object_threshold = [150 / 4000,  # front left
                                 150 / 4000,  # front right
                                 150 / 3200,  # rear top
                                 150 / 3200,  # rear bottom
                                 150 / 1000,  # left side
                                 150 / 1000,  # right side
                                 150 / 2200,  # down front
                                 150 / 2200,  # down back
                                 30 / 800]  # top
        penalization = 0
        penalization_str = ''
        # object_near
        if any(check_near_object(info["dist_sensors"],
                                 near_object_threshold)):
            logger.info(f"[{info['timestamp']}] Penalty state, ObjectNear")
            penalization -= 1.
            penalization_str += 'ObjectNear|'
        # is_collision
        if any(check_collision(info["dist_sensors"])):
            logger.info(f"[{info['timestamp']}] Penalty state, Near2Collision")
            penalization -= 2.
            penalization_str += 'Near2Collision|'
        # outside flight area
        if any(check_flight_area(info["position"], self.flight_area)):
            logger.info(f"[{info['timestamp']}] Penalty state, OutFlightArea")
            penalization -= 2.
            penalization_str = 'OutFlightArea|'
        # risk zone trespassing
        if self.sim.get_target_distance() < self.compute_risk_dist():
            logger.info(f"[{info['timestamp']}] Penalty state, InsideRiskZone")
            penalization -= 2.
            penalization_str = 'InsideRiskZone|'

        if len(penalization_str) > 0:
            info['penalization'] = penalization_str

        return penalization

    def compute_reward(self, obs, info):
        """Compute the distance-based reward.

        Compute the distance between drone and fire.
        This consider a risk_zone to 4 times the fire height as mentioned in
        Firefighter Safety Zones: A Theoretical Model Based on Radiative
        Heating, Butler, 1998.

        :param float distance_threshold: Indicate the acceptable distance
            margin before the fire's risk zone.
        """
        info['penalization'] = 'no'
        info['final'] = 'no'

        # 2 dimension considered
        if len(self.last_info.keys()) == 0:
            uav_pos_t = info['position'][:2]  # pos_t
        else:
            uav_pos_t = self.last_info['position'][:2]  # pos_t
        uav_pos_t1 = info['position'][:2]  # pos_t+1
        uav_ori_t1 = info['north_rad']  # orientation_t+1
        target_xy = self.sim.get_target_pos()[:2]

        # compute reward components
        reward = compute_vector_reward(
            target_xy, uav_pos_t, uav_pos_t1, uav_ori_t1,
            distance_target=self.distance_target,
            distance_margin=self._goal_threshold)

        if self.is_pixels:
            reward += compute_visual_reward(obs)

        # not terminal, must be avoided
        penalization = self.__compute_penalization(info)
        if penalization < 0:
            reward += penalization

        # terminal states
        discount = self.__is_final_state(info)
        if discount < 0:
            self._end = True
            reward += discount

        return reward

    def __time_limit(self):
        # time limit control
        self._episode_steps += 1
        return self._episode_steps >= self._max_episode_steps

    def perform_action(self, action):
        # perform constrained action
        constrained_action = self.sim.clip_action(action, self.flight_area)
        command = dict(disturbances=constrained_action)
        self.sim.send_data(command)
        # read new state
        return self.get_state()

    def lift_uav(self, altitude):
        diff_altitude = float('inf')
        lift_action = [0., 0., 0., self.sim.limits[1][3]]
        logger.info("Lifting the drone...")
        # wait for lift momentum
        while diff_altitude > 13.:
            _, info = self.perform_action(lift_action)
            diff_altitude = altitude - info['position'][2]  # Z axis diff
        # change vertical position
        tpos = info['position']
        tpos[2] = self.init_altitude - 0.1
        self.sim.drone_node['set_pos'](tpos)
        # wait for altitude
        while diff_altitude > 0:
            _, info = self.perform_action(lift_action)
            diff_altitude = altitude - info['position'][2]  # Z axis diff
        logger.info("Drone lifted")

        self.perform_action([0., 0., 0., 0.])  # no action

    def reset(self, seed=None, fire_cuadrant=None, **kwargs):
        """Reset episode in the Webots simulation."""
        # restart simulation
        self.seed(seed)
        self.sim.reset()
        if fire_cuadrant is None:
            self.set_fire_cuadrant(self._fire_pos)
        else:
            self.set_fire_cuadrant(fire_cuadrant)
        self.sim.play_fast()
        self.sim.sync()
        self.lift_uav(self.init_altitude)
        self.init_runtime_vars()
        self.last_state, self.last_info = self.get_state()

        return self.last_state, self.last_info

    def step(self, action):
        """Perform an action step in the simulation scene."""
        reward = 0
        # reaction time
        react_frames = self.np_random.integers(low=self._frame_inter[0],
                                               high=self._frame_inter[1],
                                               endpoint=True)
        for i in range(react_frames):
            observation, info = self.perform_action(action)
            if self._end:
                break

        # compute reward
        reward = self.compute_reward(observation, info)  # obtain reward

        # timeout limit
        if self.__time_limit():
            logger.info(f"[{info['timestamp']}] Final state, Time limit")
            self._end = True
            info['final'] = 'time_limit'

        # normalize step reward
        # reward = self.norm_reward(reward)

        self.last_state, self.last_info = observation, info

        return observation, reward, self._end, False, info

    def render(self, mode='human'):
        """Render the environment from Webots simulation."""
        if mode == 'rgb_array':
            return self.last_info['image']

        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()

            if 'image' in self.last_info.keys():
                self.viewer.imshow(self.last_info['image'])

            return self.viewer.isopen

    def close(self):
        """Close the environment and stop the simulation."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
