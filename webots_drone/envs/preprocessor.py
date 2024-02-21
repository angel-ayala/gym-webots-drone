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

from ..utils import min_max_norm


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
        rgb_obs = np.swapaxes(rgb_obs, 2, 0)
        # normalize
        rgb_obs = preprocess_pixels(rgb_obs)
    return rgb_obs


def info2emitter_vector(info):
    emitter_vector = np.zeros((4, ), dtype=np.float32)
    if info is not None:
        emitter_vector[:3] = info['emitter']['direction']  # euler angles
        emitter_vector[-1] = info['emitter']['signal_strength']  # beacon signal
    return emitter_vector


def info2obs_1d(infos, xyz_ranges, xyz_velocities):
    vector_state = info2state(infos)
    # normalize
    sensor_attitude = state2inertial(vector_state)
    sensor_attitude = preprocess_angles(sensor_attitude)
    sensor_position = state2position(vector_state)
    sensor_position = preprocess_position(sensor_position, *xyz_ranges, *xyz_velocities)
    sensor_north = vector_state[-1] / np.pi
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


def preprocess_pixels(obs):
    return obs.astype(np.float32) / 255.

def preprocess_position(obs, x_range, y_range, z_range, x_vel, y_vel, z_vel):
    # Normalize position
    obs[0] = min_max_norm(obs[0], a=-1, b=1, minx=x_range[0], maxx=x_range[1])
    obs[1] = min_max_norm(obs[1], a=-1, b=1, minx=y_range[0], maxx=y_range[1])
    obs[2] = min_max_norm(obs[2], a=-1, b=1, minx=z_range[0], maxx=z_range[1])
    # Normalize translational velocities
    obs[3] /= x_vel
    obs[4] /= y_vel
    obs[5] /= z_vel
    return obs


def preprocess_angles(obs):
    # Normalize Euler angles
    obs[0] = min_max_norm(obs[0], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    obs[1] = min_max_norm(obs[1], a=-1, b=1, minx=-np.pi/2, maxx=np.pi/2)
    obs[2] = min_max_norm(obs[2], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    # Normalize angular velocities
    obs[3] = min_max_norm(obs[3], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    obs[4] = min_max_norm(obs[4], a=-1, b=1, minx=-np.pi/2, maxx=np.pi/2)
    obs[5] = min_max_norm(obs[5], a=-1, b=1, minx=-np.pi, maxx=np.pi)
    return obs


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


class MultiModalObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, shape1=(3, 84, 84), shape2=(22, ),
                 frame_stack=1):
        super().__init__(env)
        self.rgb_obs = spaces.Box(low=0, high=1, shape=shape1,
                                  dtype=np.float32)
        self.vector_obs = spaces.Box(low=float('-inf'), high=float('inf'),
                                     shape=shape2, dtype=np.float32)
        self.frame_stack = frame_stack
        if frame_stack > 1:
            env.observation_space = self.rgb_obs
            self.env_rgb = gym.wrappers.FrameStack(env, num_stack=frame_stack)
            self.rgb_obs = self.env_rgb.observation_space
            env.observation_space = self.vector_obs
            self.env_vector = gym.wrappers.FrameStack(env, num_stack=frame_stack)
            self.vector_obs = self.env_vector.observation_space

        self.observation_space = spaces.Tuple((self.rgb_obs, self.vector_obs))

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        # order sensors by dimension and split
        obs_2d = info2image(infos, output_size=self.rgb_obs.shape[-1])
        obs_1d = info2obs_1d(infos)
        new_obs = (obs_2d, obs_1d)
        if self.frame_stack > 1:
            self.env_rgb.frames.append(obs_2d)
            self.env_vector.frames.append(obs_1d)
            new_obs = (self.env_rgb.observation(None),
                       self.env_vector.observation(None))

        return new_obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)
        obs_2d = info2image(info, output_size=self.rgb_obs.shape[-1])
        obs_1d = info2obs_1d(info)
        new_obs = (obs_2d, obs_1d)
        if self.frame_stack > 1:
            for _ in range(self.num_stack):
                self.env_rgb.frames.append(obs_2d)
                self.env_vector.frames.append(obs_1d)
            new_obs = (self.env_rgb.observation(None),
                       self.env_vector.observation(None))

        return new_obs, info
