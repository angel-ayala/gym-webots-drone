#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:17:26 2024

@author: Angel Ayala
"""

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler

from webots_drone.stack import ObservationStack


UAV_DATA = ['imu', 'gyro', 'gps', 'gps_vel', 'north', 'dist_sensors',
            'motors', 'target_sensors']


def seconds2steps(seconds, frame_skip, step_time):
    total_step_time = frame_skip * step_time
    return int(seconds * 1000 / total_step_time)


def info2state(info):
    vector_state = np.zeros((12, ), dtype=np.float32)
    if info is not None:
        vector_state[:3] = info['position']  # world coordinates
        vector_state[3:6] = info['orientation']  # euler angles
        vector_state[6:9] = info['speed']
        vector_state[9:12] = info['angular_velocity']
    return vector_state


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


class CustomVectorObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, uav_data=UAV_DATA,
                 target_dist=False, target_pos=False, target_dim=False,
                 add_action=False, angles_range=[np.pi, np.pi / 2., np.pi],
                 avel_range=[np.pi, np.pi / 2., np.pi],
                 speed_range=[4., 4., 1.], n_dist_sensors=9, norm_obs=False):
        super().__init__(env)
        obs_elems = 0
        obs_high_limits = []
        obs_low_limits = []
        self.uav_data = uav_data
        if 'imu' in uav_data:
            obs_elems += 3
            obs_high_limits.extend(angles_range)
            obs_low_limits.extend([-v for v in angles_range])
        if 'gyro' in uav_data:
            obs_elems += 3
            obs_high_limits.extend(avel_range)
            obs_low_limits.extend([-v for v in avel_range])
        if 'gps' in uav_data:
            obs_elems += 3
            obs_high_limits.extend(self.env.unwrapped.flight_area[1])
            obs_low_limits.extend(self.env.unwrapped.flight_area[0])
        if 'gps_vel' in uav_data:
            obs_elems += 3
            obs_high_limits.extend(speed_range)
            obs_low_limits.extend([-v for v in speed_range])
        if 'north' in uav_data:
            obs_elems += 1
            obs_high_limits.append(angles_range[2])
            obs_low_limits.append(-angles_range[2])
        if 'dist_sensors' in uav_data:
            obs_elems += n_dist_sensors
            obs_high_limits.extend([1. for _ in range(n_dist_sensors)])
            obs_low_limits.extend([0. for _ in range(n_dist_sensors)])
        if 'target_sensors' in uav_data:
            obs_elems += 6
            obs_high_limits.extend([1. for _ in range(6)])
            obs_low_limits.extend([0. for _ in range(6)])
        if 'motors' in uav_data:
            obs_elems += 4
            obs_high_limits.extend([600. for _ in range(4)])
            obs_low_limits.extend([0. for _ in range(4)])

        self.target_pos = target_pos
        if target_pos:
            obs_elems += 3
            obs_high_limits.extend(self.env.unwrapped.flight_area[1])
            obs_low_limits.extend(self.env.unwrapped.flight_area[0])
        self.target_dim = target_dim
        if target_dim:
            obs_elems += 2
            obs_high_limits.extend([10., 10.])
            obs_low_limits.extend([0., 0.])
        self.target_dist = target_dist
        if target_dist:
            obs_elems += 4
            obs_high_limits.extend([10., 10., 10., 1.])
            obs_low_limits.extend([0., 0., 0., -1.])

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

        obs_shape = (obs_elems, )
        self.observation_space = spaces.Box(low=np.asarray(obs_low_limits),
                                            high=np.asarray(obs_high_limits),
                                            shape=obs_shape, dtype=np.float32)
        self.norm_obs = norm_obs
        if norm_obs:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit([self.observation_space.low,
                             self.observation_space.high])

    def observation(self, info, action, norm_obs=False):
        new_obs = []
        if 'imu' in self.uav_data:
            new_obs.extend(info['orientation'])
        if 'gyro' in self.uav_data:
            new_obs.extend(info['angular_velocity'])
        if 'gps' in self.uav_data:
            new_obs.extend(info['position'])
        if 'gps_vel' in self.uav_data:
            new_obs.extend(info['speed'])
        if 'north' in self.uav_data:
            new_obs.append(info['north_rad'])
        if 'dist_sensors' in self.uav_data:
            new_obs.extend(info['dist_sensors'])
        if 'target_sensors' in self.uav_data:
            new_obs.extend(self.env.unwrapped.vtarget.get_sensor_readings(
                info['position'], info['north_rad']))
        if 'motors' in self.uav_data:
            new_obs.extend(info['motors_vel'])

        if self.target_pos:
            new_obs.extend(self.env.unwrapped.vtarget.position)
        if self.target_dim:
            new_obs.extend(self.env.unwrapped.vtarget.dimension)
        if self.target_dist:
            new_obs.extend(np.subtract(
                self.env.unwrapped.vtarget.position, info['position']))
        # append action t-1
        if self.add_action:
            if len(self.env.action_space.shape) > 0:
                action_max = self.env.action_limits[1][:self.action_vars]
                new_obs.extend(action / action_max)
            else:
                new_obs.append(action)
        new_obs = np.asarray(new_obs)
        if norm_obs:
            new_obs = self.scaler.transform([new_obs])
        return new_obs

    def step(self, action):
        obs, rews, terminateds, truncateds, info = self.env.step(action)
        # adding target vector, expecting info2obs_1d
        new_obs = self.observation(info, action, self.norm_obs)
        return new_obs, rews, terminateds, truncateds, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)
        new_obs = self.observation(info, np.zeros(self.action_space.shape),
                                   self.norm_obs)
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
    def __init__(self, env: gym.Env, uav_data=UAV_DATA,
            frame_stack=1, target_dist=False, target_pos=False,
            target_dim=False, add_action=False,
            angles_range=[np.pi, np.pi / 2., np.pi],
            avel_range=[np.pi, np.pi / 2., np.pi], speed_range=[4., 4., 1.],
            norm_obs=False):
        super().__init__(env)
        self.pixel_space = env.observation_space
        self.vector_obs = CustomVectorObservation(
            env, uav_data=uav_data, target_dist=target_dist,
            target_pos=target_pos, target_dim=target_dim,
            add_action=add_action, angles_range=angles_range,
            avel_range=avel_range, speed_range=speed_range, norm_obs=norm_obs)
        self.vector_space = self.vector_obs.observation_space
        self.frame_stack = frame_stack

        if frame_stack > 1:
            self.pixel_obs = ObservationStack(env, k=frame_stack)
            self.pixel_space = self.pixel_obs.observation_space
            self.vector_stack = ObservationStack(self.vector_obs, k=frame_stack)
            self.vector_space = self.vector_stack.observation_space

        self.observation_space = spaces.Dict({'vector': self.vector_space,
                                              'pixel': self.pixel_space})

    def get_state(self, action):
        """Process the environment to get a state."""
        _, state_data = self.env.unwrapped.get_state()
        # order sensors by dimension and split
        state_2d = self.env.unwrapped.get_observation_2d(state_data)
        # state_1d = self.env.unwrapped.get_observation_1d(state_data)
        state_1d = self.vector_obs.observation(state_data, action)
        return {'vector': state_1d, 'pixel': state_2d}

    def step(self, action):
        _, rews, terminateds, truncateds, info = self.env.step(action)
        new_obs = self.get_state(action)

        if self.frame_stack > 1:
            self.pixel_obs.frames.append(new_obs['pixel'])
            self.vector_stack.frames.append(new_obs['vector'][np.newaxis, ...])
            new_obs = {'vector': self.vector_stack.observation(None),
                       'pixel': self.pixel_obs.observation(None)}

        return new_obs, rews, terminateds, truncateds, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        _, info = self.env.reset(**kwargs)
        new_obs = self.get_state(np.zeros(self.action_space.shape))
        if self.frame_stack > 1:
            for _ in range(self.frame_stack):
                self.pixel_obs.frames.append(new_obs['pixel'])
                self.vector_stack.frames.append(new_obs['vector'][np.newaxis, ...])
            new_obs = {'vector': self.vector_stack.observation(None),
                       'pixel': self.pixel_obs.observation(None)}

        return new_obs, info
