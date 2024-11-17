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
from webots_drone.utils import compute_distance
from webots_drone.utils import check_collision
from webots_drone.utils import check_flight_area
from webots_drone.utils import check_flipped
from webots_drone.utils import check_near_object
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
                 target_pos=2,
                 target_dim=[7., 3.5],
                 is_pixels=True,
                 zone_steps=10):

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
        self._target_pos = target_pos
        self._target_dim = target_dim
        self.init_altitude = init_altitude
        self.viewer = None
        # flight_area and target discrete position
        self.flight_area = self.sim.get_flight_area(altitude_limits)
        self.sample_quadrants = list()
        self.max_distance = compute_distance(*self.flight_area)
        self.quadrants = self.create_quadrants()

        # self.reward_limits = [-2. - (2.21 * self._frame_inter[0]),
        #                       3.21 * self._frame_inter[1]]
        self.zone_steps = zone_steps if zone_steps > 0 else float('inf')

        # virtualTarget
        self.vtarget = self.create_target(target_dim)

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

    def create_target(self, dimension=None):
        # virtualTarget
        from webots_drone.target import VirtualTarget
        return VirtualTarget(dimension=dimension,
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
        self._zone_flags = [False, False, False]  # zone control flags
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

    def get_target_quadrant(self, quadrant=None):
        if quadrant is None:
            # random shuffle quadrant with no reposition
            if len(self.sample_quadrants) == 0:
                self.sample_quadrants = list(range(len(self.quadrants)))
                self.np_random.shuffle(self.sample_quadrants)
            quadrant = self.sample_quadrants.pop(0)
        return self.quadrants[quadrant]

    def set_target(self, position=None, dimension=None):
        # update position
        tpos = self._target_pos if position is None else position
        if type(tpos) is int or tpos is None:
            tpos = self.get_target_quadrant(tpos)

        # ensure fire position inside the flight_area
        tpos[0] = np.clip(tpos[0], *self.flight_area[:, 0])
        tpos[1] = np.clip(tpos[1], *self.flight_area[:, 1])
        if self.vtarget.is_3d:
            tpos[2] = np.clip(tpos[2], *self.flight_area[:, 2])

        self.vtarget.set_position(tpos)

        # update dimension
        tdim = self._target_dim if dimension is None else dimension
        self.vtarget.set_dimension(*tdim)
        logger.info(str(self.vtarget))

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
        """Compute the target distance considering the Risk distance, the Goal
        threshold and the Vehicle's radius."""
        return self.vtarget.get_risk_distance(self._goal_threshold / 2.) \
            + self.sim.vehicle_dim[1]

    def update_no_action_counter(self, position, pos_thr=0.003):
        if len(self.last_info.keys()) == 0:
            return False
        if check_same_position(position, self.last_info['position'],
                               thr=pos_thr) and not self._zone_flags[1]:
            self._no_action_steps += 1
        else:
            self._no_action_steps = 0

    @property
    def no_action_limit(self):
        return self._no_action_steps >= self._max_no_action_steps

    def update_zone_steps_counter(self):
        if self._zone_flags[1]:  # in-zone
            self._in_zone_steps += 1
        else:
            self._in_zone_steps = 0

    @property
    def zone_steps_limit(self):
        return self._in_zone_steps >= self.zone_steps

    def __is_final_state(self, info):
        is_final = False
        # no action limit
        if self.no_action_limit:
            logger.info(f"[{info['timestamp']}] Final state, Same position")
            info['final'] = 'NoAction'
            is_final = True
        # is_flipped
        elif check_flipped(info["orientation"], info["dist_sensors"]):
            logger.info(f"[{info['timestamp']}] Final state, Flipped")
            info['final'] = 'Flipped'
            is_final = True
        elif self.vtarget.get_distance(info['position']) > self.max_distance:
            logger.info(f"[{info['timestamp']}] Final state, MaxDistance")
            info['final'] = 'MaxDistance'
            is_final = True
        elif self._zone_flags[0]:
            logger.info(f"[{info['timestamp']}] Final state, InsideRiskZone")
            info['final'] = 'InsideRiskZone'
            is_final = True

        return is_final

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
        if self._zone_flags[0]:
            logger.info(f"[{info['timestamp']}] Penalty state, InsideRiskZone")
            penalization -= 2.
            penalization_str += 'InsideRiskZone|'

        if len(penalization_str) > 0:
            info['penalization'] = penalization_str

        return penalization

    def get_uav_zone(self, position):
        distance2target = self.vtarget.get_distance(position)
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

        # 2 dimension considered
        if not is_3d:
            uav_pos_t[2] = uav_pos_t1[2] = self.vtarget.position[2]

        # compute reward components
        reward = compute_vector_reward(
            self.vtarget, uav_pos_t, uav_pos_t1, uav_ori_t1,
            distance_target=self.distance_target,
            distance_margin=self._goal_threshold,
            vel_factor=vel_factor, pos_thr=pos_thr)

        # if self.is_pixels:
        #     reward += compute_visual_reward(obs)

        # no action and zone steps update
        self.update_no_action_counter(uav_pos_t1)
        self.update_zone_steps_counter()

        # not terminal, must be avoided
        penalization = self.__compute_penalization(info)
        if penalization < 0:
            reward += penalization

        return reward

    def __time_limit(self):
        # time limit control
        self._episode_steps += 1
        return self._episode_steps >= self._max_episode_steps

    def constraint_action(self, action, info):
        return constrained_action(action, info['position'], info['north_rad'],
                                  self.flight_area)

    def perform_action(self, action):
        # perform constrained action
        info = self.last_info
        c_action = self.constraint_action(action, info)
        self.sim.send_action(c_action)
        # read new state
        observation, info = self.get_state()
        self._zone_flags = self.get_uav_zone(info['position'])
        # terminal states
        self._end = self.__is_final_state(info)

        return observation, info

    def lift_uav(self):
        logger.info("Drone's taking off...")
        self.sim.take_off(self.init_altitude)
        logger.info("Drone ready!")

        self.perform_action([0., 0., 0., 0.])  # no action to update state only

    def reset(self, seed=None, target_pos=None, target_dim=None, **kwargs):
        """Reset episode in the Webots simulation."""
        # restart simulation
        self.seed(seed)
        self.sim.reset()
        self.sim.play_fast()
        self.sim.sync()
        self.last_state, self.last_info = self.get_state()
        self.set_target(position=target_pos, dimension=target_dim)
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
        if self.zone_steps_limit:
            logger.info(f"[{info['timestamp']}] Final state, Goal reached")
            truncated = True
            info['final'] = 'GoalFound'

        # penalize end states
        if self._end:
            reward -= 2.

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
