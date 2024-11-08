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

from webots_drone.reward import compute_vector_reward
from webots_drone.reward import compute_visual_reward
from webots_drone.utils import check_collision
from webots_drone.utils import check_flight_area
from webots_drone.utils import check_flipped
from webots_drone.utils import check_near_object
# from webots_drone.utils import min_max_norm
from webots_drone.utils import check_same_position
from webots_drone.utils import check_target_distance
from webots_drone.utils import constrained_action

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
                 is_pixels=True,
                 zone_steps=0):

        self.init_sim()
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
        self._max_no_action_steps = seconds2steps(
            max_no_action_seconds, frame_skip, self.sim.timestep)
        self.set_reaction_intervals(frame_skip)

        self._goal_threshold = goal_threshold
        self._fire_pos = fire_pos
        self._fire_dim = fire_dim
        self.init_altitude = init_altitude
        self.viewer = None
        # flight_area and target discrete position
        self.flight_area = self.sim.get_flight_area(altitude_limits)
        self.sample_quadrants = list()
        self.quadrants = self.create_quadrants()

        # self.reward_limits = [-2. - (2.21 * self._frame_inter[0]),
        #                       3.21 * self._frame_inter[1]]
        self.zone_steps = zone_steps if zone_steps > 0 else float('inf')

        # virtualTarget
        self.vtarget = self.create_target(fire_pos, fire_dim)

    def init_sim(self):
        from webots_drone import WebotsSimulation
        # Simulation controller
        logger.info('Checking Webots connection...')
        self.sim = WebotsSimulation()
        logger.info('Connected to Webots')

    def create_quadrants(self):
        quadrants = np.array(
            [(self.flight_area[0][0], self.flight_area[1][1]),
             (self.flight_area[1][0], self.flight_area[1][1]),
             (self.flight_area[1][0], self.flight_area[0][1]),
             (self.flight_area[0][0], self.flight_area[0][1])])
        quadrants /= 2.
        return quadrants

    def set_reaction_intervals(self, frame_skip):
        self._frame_inter = [frame_skip - 5., frame_skip + 5.]

    def create_target(self, position=None, dimension=None):
        # virtualTarget
        from webots_drone.target import VirtualTarget
        if type(position) is int:
            position = self.quadrants[position]
        return VirtualTarget(position=position, dimension=dimension,
                             webots_node=self.sim.target_node)

    @property
    def action_limits(self):
        return self.sim.get_control_ranges()

    # def norm_reward(self, reward):
    #     reward = min_max_norm(reward - 1e-8,  # avoids zero values
    #                           -1., 2., self.reward_limits[0],
    #                           self.reward_limits[1])
    #     return reward


    def init_runtime_vars(self):
        self._episode_steps = 0  # time limit control
        self._no_action_steps = 0  # no action control
        self._in_zone_steps = 0  # time inside zone control
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
        self.vtarget.seed(seed)
        return [seed1, seed2]

    def set_fire_position(self, position, noise_prob=0.):
        offset = 0
        if self.np_random.random() < noise_prob:
            offset = self.np_random.random()
        fire_pos = position + offset
        logger.info(f"Starting fire in position {position} with offset = {offset}.")
        self.vtarget.set_dimension(*self._fire_dim)
        new_fire_pos = self.vtarget.set_position(self.flight_area, fire_pos)

        # avoid to the fire appears near the drone's initial position
        must_do = 0
        dist_threshold = self._goal_threshold * 1.5
        directions = [(-1, 1), (1, 1),
                      (1, -1), (-1, -1)]
        while (self.vtarget.get_distance(self.last_info['position'])
               <= self.vtarget.get_risk_distance(dist_threshold)):
            # randomize position offset
            offset = self.np_random.uniform(0.1, 1.)
            new_fire_pos[0] += offset * directions[must_do % 4][0]
            new_fire_pos[1] += offset * directions[must_do % 4][1]
            must_do += 1
            self.set_fire_position(new_fire_pos)

    def set_fire_quadrant(self, quadrant=None, noise_prob=0.):
        if quadrant is None:
            # random shuffle quadrant with no reposition
            if len(self.sample_quadrants) == 0:
                self.sample_quadrants = list(range(len(self.quadrants)))
                self.np_random.shuffle(self.sample_quadrants)
            quadrant = self.sample_quadrants.pop(0)
        self.set_fire_position(self.quadrants[quadrant], noise_prob=noise_prob)

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
        state_data['target_position'] = self.vtarget.position
        state_data['target_dim'] = self.vtarget.dimension

        if self.is_pixels:
            state = self.get_observation_2d(state_data)
        else:
            state = self.get_observation_1d(state_data)

        return state, state_data

    @property
    def distance_target(self):
        return self.vtarget.get_risk_distance(self._goal_threshold / 2.)

    def no_action_limit(self, position, pos_thr=0.003):
        if len(self.last_info.keys()) == 0:
            return False
        if check_same_position(position, self.last_info['position'],
                               thr=pos_thr):
            self._no_action_steps += 1
        else:
            self._no_action_steps = 0
        return self._no_action_steps >= self._max_no_action_steps

    def __is_final_state(self, info, zones):
        discount = 0
        # no action limit
        if self.no_action_limit(info["position"]) and not zones[1]:
            logger.info(f"[{info['timestamp']}] Final state, Same position")
            discount -= 2.
            info['final'] = 'No Action'
        # is_flipped
        elif check_flipped(info["orientation"], info["dist_sensors"]):
            logger.info(f"[{info['timestamp']}] Final state, Flipped")
            discount -= 2.
            info['final'] = 'Flipped'

        return discount

    def __compute_penalization(self, info, zones):
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
        if zones[0]:
            logger.info(f"[{info['timestamp']}] Penalty state, InsideRiskZone")
            penalization -= 2.
            penalization_str += 'InsideRiskZone|'

        if len(penalization_str) > 0:
            info['penalization'] = penalization_str

        return penalization

    def get_uav_zone(self, info):
        distance2target = self.vtarget.get_distance(info['position'])
        zones = check_target_distance(distance2target,
                                      self.distance_target,
                                      self._goal_threshold)
        return zones

    def compute_reward(self, obs, info, is_3d=False, vel_factor=0.035,
                       pos_thr=0.003):
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

        # capture essential info
        if len(self.last_info.keys()) == 0:
            uav_pos_t = info['position']  # pos_t
        else:
            uav_pos_t = self.last_info['position']  # pos_t
        uav_pos_t1 = info['position']  # pos_t+1
        uav_ori_t1 = info['north_rad']  # orientation_t+1
        target_xy = self.vtarget.position

        # 2 dimension considered
        if not is_3d:
            uav_pos_t[2] = uav_pos_t1[2] = target_xy[2]

        # compute reward components
        reward = compute_vector_reward(
            target_xy, uav_pos_t, uav_pos_t1, uav_ori_t1,
            distance_target=self.distance_target,
            distance_margin=self._goal_threshold,
            vel_factor=vel_factor, pos_thr=pos_thr)

        # if self.is_pixels:
        #     reward += compute_visual_reward(obs)
        zones = self.get_uav_zone(info)

        # allow no action inside zone
        if zones[1]:
            self._in_zone_steps += 1
        else:
            self._in_zone_steps = 0

        # not terminal, must be avoided
        penalization = self.__compute_penalization(info, zones)
        if penalization < 0:
            reward += penalization

        # terminal states
        discount = self.__is_final_state(info, zones)
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
        info = self.last_info
        c_action = constrained_action(action, info['position'],
                                      info['north_rad'], self.flight_area)
        self.sim.send_action(c_action)
        # read new state
        observation, info = self.get_state()
        zones = self.get_uav_zone(info)
        # ensure no enter risk_zone
        if zones[0]:
            logger.info(f"[{info['timestamp']}] Final state, InsideRiskZone")
            info['final'] = 'InsideRiskZone'
            self._end = True

        return observation, info

    def lift_uav(self):
        diff_altitude = float('inf')
        lift_action = [0., 0., 0., self.sim.limits[1][3]]
        logger.info("Lifting the drone...")
        # wait for lift momentum
        while diff_altitude > 13.:
            _, info = self.perform_action(lift_action)
            diff_altitude = self.init_altitude - info['position'][2]  # Z axis diff
        # change vertical position
        tpos = info['position']
        tpos[2] = self.init_altitude - 0.1
        self.sim.drone_node['set_pos'](tpos)
        # wait for altitude
        while diff_altitude > 0:
            _, info = self.perform_action(lift_action)
            diff_altitude = self.init_altitude - info['position'][2]  # Z axis diff
        logger.info("Drone lifted")

        self.perform_action([0., 0., 0., 0.])  # no action

    def reset(self, seed=None, fire_quadrant=None, **kwargs):
        """Reset episode in the Webots simulation."""
        # restart simulation
        self.seed(seed)
        self.sim.reset()
        self.sim.play_fast()
        self.sim.sync()
        self.last_state, self.last_info = self.get_state()
        fquadrant = self._fire_pos if fire_quadrant is None else fire_quadrant
        self.set_fire_quadrant(fquadrant)
        self.lift_uav()
        self.init_runtime_vars()
        self.last_state, self.last_info = self.get_state()

        return self.last_state, self.last_info

    def step(self, action):
        """Perform an action step in the simulation scene."""
        reward = 0
        truncated = False
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
            truncated = True
            info['final'] = 'TimeLimit'
        # goal state
        if self._in_zone_steps >= self.zone_steps:
            logger.info(f"[{info['timestamp']}] Final state, Goal reached")
            truncated = True
            info['final'] = 'GoalFound'

        # normalize step reward
        # reward = self.norm_reward(reward)

        self.last_state, self.last_info = observation, info

        return observation, reward, self._end, truncated, info

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
