#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:17:26 2024

@author: Angel Ayala
"""

import cv2
import numpy as np
import gym
from gym import spaces

from webots_drone.utils import min_max_norm
from webots_drone.utils import flight_area_norm_position
from webots_drone.utils import compute_target_orientation
from webots_drone.utils import compute_distance
from webots_drone.stack import ObservationStack


def seconds2steps(seconds, frame_skip, step_time):
    total_step_time = frame_skip * step_time
    return int(seconds * 1000 / total_step_time)


def info2state(info):
    vector_state = np.zeros((13, ), dtype=np.float32)
    if info is not None:
        vector_state[:3] = info['position']  # world coordinates
        vector_state[3:6] = info['orientation']  # euler angles
        vector_state[6:9] = info['speed']
        vector_state[9:12] = info['angular_velocity']
        vector_state[-1] = info['north_rad']
    return vector_state


def state2position(vector_state):
    inertial_state = np.concatenate((vector_state[:3],
                                     vector_state[6:9]), dtype=np.float32)
    return inertial_state


def state2inertial(vector_state):
    position = np.concatenate((vector_state[3:6],
                               vector_state[9:12]), dtype=np.float32)
    return position


def info2distance(info):
    distance_sensors = np.zeros((9, ), dtype=np.float32)
    if info is not None:
        distance_sensors = np.array(info["dist_sensors"], dtype=np.float32)
    return distance_sensors


def info2image(info, output_size):
    rgb_obs = None
    if info is not None:
        rgb_obs = info["image"].copy()[:, :, [2, 1, 0]]  # RGB copy
        # crop square
        rgb_obs = crop_from_center(rgb_obs)
        # resize
        rgb_obs = cv2.resize(rgb_obs, (output_size, output_size),
                             interpolation=cv2.INTER_AREA)
        # channel first
        rgb_obs = np.transpose(rgb_obs, axes=(2, 0, 1))
    return rgb_obs


def info2emitter_vector(info):
    emitter_vector = np.zeros((4, ), dtype=np.float32)
    if info is not None:
        emitter_vector[:3] = info['emitter']['direction']  # euler angles
        emitter_vector[-1] = info['emitter']['signal_strength']  # beacon signal
    return emitter_vector


def info2obs_1d(infos):
    vector_state = info2state(infos)
    # normalize
    sensor_attitude = state2inertial(vector_state)
    sensor_position = state2position(vector_state)
    sensor_north = vector_state[-1]
    sensor_distance = info2distance(infos)
    # sensor_emitter = info2emitter_vector(infos)
    obs_1d = np.hstack((sensor_attitude, sensor_position,
                        sensor_north, sensor_distance), dtype=np.float32)
    return obs_1d


def preprocess_orientation(orientation):
    # Convert from [-pi, pi] to [0, 2pi]
    if orientation < 0:
        orientation += 2 * np.pi
    return orientation


def normalize_pixels(obs):
    return obs / 255.


def normalize_position(obs, x_range, y_range, z_range, x_vel, y_vel, z_vel):
    # Normalize position
    obs[0] = min_max_norm(obs[0], a=-1, b=1, minx=x_range[0], maxx=x_range[1])
    obs[1] = min_max_norm(obs[1], a=-1, b=1, minx=y_range[0], maxx=y_range[1])
    obs[2] = min_max_norm(obs[2], a=-1, b=1, minx=z_range[0], maxx=z_range[1])
    # Normalize translational velocities
    obs[3] /= x_vel
    obs[4] /= y_vel
    obs[5] /= z_vel
    return obs


def normalize_angles(obs):
    # Normalize Euler angles
    obs[0] = min_max_norm(obs[0], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    obs[1] = min_max_norm(obs[1], a=-1, b=1, minx=-np.pi/2, maxx=np.pi/2)
    obs[2] = min_max_norm(obs[2], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    # Normalize angular velocities
    obs[3] = min_max_norm(obs[3], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    obs[4] = min_max_norm(obs[4], a=-1, b=1, minx=-np.pi/2, maxx=np.pi/2)
    obs[5] = min_max_norm(obs[5], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    return obs


def normalize_vector(vector, xyz_range, xyz_vel):
    norm_vector = vector.copy()
    norm_vector[:6] = normalize_angles(norm_vector[:6])
    norm_vector[6:12] = normalize_position(norm_vector[6:12], *xyz_range, *xyz_vel)
    norm_vector[12] /= np.pi
    return norm_vector


def crop_from_center(img):
    """center crop image."""
    # make it square from center
    h, w, _ = img.shape
    hheight = h // 2
    hcenter = w // 2
    center_idx = (hcenter - hheight, hcenter + hheight)
    result = np.asarray(img[:, center_idx[0]:center_idx[1], :3],
                        dtype=img.dtype)
    return result


def info2target_position(info):
    target_pos = info['target_position']
    return target_pos


def info2target_distance(info):
    # distance
    uav_pos = info['position']
    target_pos = info['target_position']
    target_dist = uav_pos - target_pos
    # orientation
    target_ori = compute_target_orientation(uav_pos[:2], target_pos[:2])
    ori_diff_norm = np.cos(target_ori - info['north_rad'])
    return target_dist, ori_diff_norm


def info2target_dim(info):
    target_dim = np.array(info['target_dim'])  # / 10.
    return target_dim.tolist()


class CustomVectorObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, uav_data=['imu', 'gyro', 'gps', 'gps_vel',
                                               'north', 'dist_sensors'],
                 target_dist=False, target_pos=False, target_dim=False,
                 add_action=False):
        super().__init__(env)
        obs_elems = 0
        obs_high_limits = []
        obs_low_limits = []
        self.uav_data = uav_data
        if 'imu' in uav_data:
            obs_elems += 3
            obs_high_limits.extend([np.pi, np.pi / 2., np.pi])
            obs_low_limits.extend([-np.pi, -np.pi / 2., -np.pi])
        if 'gyro' in uav_data:
            obs_elems += 3
            obs_high_limits.extend([np.pi, np.pi / 2., np.pi])
            obs_low_limits.extend([-np.pi, -np.pi / 2., -np.pi])
        if 'gps' in uav_data:
            obs_elems += 3
            obs_high_limits.extend(self.env.flight_area[1])
            obs_low_limits.extend(self.env.flight_area[0])
        if 'gps_vel' in uav_data:
            obs_elems += 3
            obs_high_limits.extend([4., 4., 1.])
            obs_low_limits.extend([-4., -4., -1.])
        if 'north' in uav_data:
            obs_elems += 1
            obs_high_limits.append(np.pi)
            obs_low_limits.append(-np.pi)
        if 'dist_sensors' in uav_data:
            obs_elems += 9
            obs_high_limits.extend([1. for _ in range(9)])
            obs_low_limits.extend([0. for _ in range(9)])

        self.target_dist = target_dist
        if target_dist:
            obs_elems += 4
            obs_high_limits.extend([10., 10., 10., 1.])
            obs_low_limits.extend([0., 0., 0., -1.])
        self.target_pos = target_pos
        if target_pos:
            obs_elems += 3
            obs_high_limits.extend(self.env.flight_area[1])
            obs_low_limits.extend(self.env.flight_area[0])
        self.target_dim = target_dim
        if target_dim:
            obs_elems += 2
            obs_high_limits.extend([10., 10.])
            obs_low_limits.extend([0., 0.])

        self.add_action = add_action
        if add_action:
            if len(self.env.action_space.shape) > 0:
                self.action_vars = self.env.action_space.shape[-1]
                obs_high_limits.extend(self.env.action_space.high)
                obs_low_limits.extend(self.env.action_space.low)
            else:
                self.action_vars = 1
                obs_high_limits.append(self.env.action_space.n)
                obs_low_limits.append(0)
            obs_elems += self.action_vars

        obs_shape = np.asarray(self.env.observation_space.shape)
        obs_shape[-1] = obs_elems
        self.observation_space = spaces.Box(low=np.asarray(obs_low_limits),
                                            high=np.asarray(obs_high_limits),
                                            shape=obs_shape, dtype=np.float32)

    def observation(self, obs, info, action):
        new_obs = list()
        if 'imu' in self.uav_data:
            new_obs.extend(obs[:3])
        if 'gyro' in self.uav_data:
            new_obs.extend(obs[3:6])
        if 'gps' in self.uav_data:
            new_obs.extend(obs[6:9])
        if 'gps_vel' in self.uav_data:
            new_obs.extend(obs[9:12])
        if 'north' in self.uav_data:
            new_obs.append(obs[12])

        if self.target_dist:
            new_obs.extend(info2target_distance(info))
        if self.target_pos:
            new_obs.extend(info2target_position(info))
        if self.target_dim:
            new_obs.extend(info2target_dim(info))
        # append action v_t-1
        if self.add_action:
            if len(self.env.action_space.shape) > 0:
                action_max = self.env.action_limits[1][:self.action_vars]
                new_obs.extend(action / action_max)
            else:
                new_obs.append(action)
        # append sensors at the end
        if 'dist_sensors' in self.uav_data:
            new_obs.extend(obs[13:])
        # print('new_obs', len(new_obs), new_obs)
        return np.array(new_obs)

    def step(self, action):
        obs, rews, terminateds, truncateds, info = self.env.step(action)
        # adding target vector, expecting info2obs_1d
        new_obs = self.observation(obs, info, action)
        return new_obs, rews, terminateds, truncateds, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)
        new_obs = self.observation(obs, info, np.zeros(self.action_space.shape))
        return new_obs, info


class ReducedActionSpace(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        control_limits = env.action_limits[:, :3]
        self.action_space = spaces.Box(low=control_limits[0],
                                       high=control_limits[1],
                                       shape=(control_limits.shape[-1], ),
                                       dtype=np.float32)
    def step(self, action):
        """Do an action step inside the Webots simulator."""
        mapped_action = np.hstack((action, [0]))
        mapped_action = np.clip(mapped_action, *self.env.action_limits)

        return self.env.step(mapped_action)


class MultiModalObservation(gym.Wrapper):
    # FIXME
    def __init__(self, env: gym.Env, shape1=(3, 84, 84), shape2=(22, ),
                 frame_stack=1, add_target=False, with_target_dim=False):
        super().__init__(env)
        self.rgb_obs = spaces.Box(low=0, high=1, shape=shape1,
                                  dtype=np.float32)
        self.vector_obs = spaces.Box(low=float('-inf'), high=float('inf'),
                                     shape=shape2, dtype=np.float32)
        self.frame_stack = frame_stack
        self.add_target = add_target
        self.with_target_dim = with_target_dim
        if add_target:
            obs_shape = np.asarray(shape2) + 3
            if with_target_dim:
                obs_shape += 2
            obs_shape += self.env.action_space.shape[-1] if len(self.env.action_space.shape) > 0 else 1
            self.vector_obs = spaces.Box(low=float('-inf'), high=float('inf'),
                                         shape=obs_shape, dtype=np.float32)

        if frame_stack > 1:
            env.observation_space = self.rgb_obs
            self.env_rgb = ObservationStack(env, k=frame_stack)
            self.rgb_obs = self.env_rgb.observation_space
            env.observation_space = self.vector_obs
            self.env_vector = ObservationStack(env, k=frame_stack)
            self.vector_obs = self.env_vector.observation_space

        self.observation_space = spaces.Tuple((self.rgb_obs, self.vector_obs))

    def add_1d_target(self, obs, info):
        return append_target_position(obs, info, self.env.flight_area)

    def get_state(self):
        """Process the environment to get a state."""
        state_data = self.env.sim.get_data()
        # order sensors by dimension and split
        state_2d = self.env.get_observation_2d(state_data)
        state_1d = self.env.get_observation_1d(state_data)
        if self.add_target:
            state_1d = self.add_1d_target(state_1d, state_data)
        return state_2d, state_1d

    def step(self, action):
        _, rews, terminateds, truncateds, info = self.env.step(action)
        new_obs = self.get_state()
        
        if self.frame_stack > 1:
            self.env_rgb.frames.append(new_obs[0])
            self.env_vector.frames.append(new_obs[1][np.newaxis, ...])
            new_obs = (self.env_rgb.observation(None),
                       self.env_vector.observation(None))

        return new_obs, rews, terminateds, truncateds, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        _, info = self.env.reset(**kwargs)
        new_obs = self.get_state()
        # append action v_t-1
        new_obs[1] = append_action(new_obs[1], np.zeros_like(self.env.action_space.shape))
        if self.frame_stack > 1:
            for _ in range(self.frame_stack):
                self.env_rgb.frames.append(new_obs[0])
                self.env_vector.frames.append(new_obs[1][np.newaxis, ...])
            new_obs = (self.env_rgb.observation(None),
                       self.env_vector.observation(None))

        return new_obs, info
