#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:11:52 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import numpy as np
from gym import spaces

from webots_drone.envs import DroneEnvContinuous


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape((len(targets), nb_classes))


class DroneEnvDiscrete(DroneEnvContinuous):
    """Gym enviroment to control the Fire scenario in the Webots simulator."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 30
    }

    def __init__(self, time_limit=7500,  # 1 min
                 max_no_action_steps=625,  # 5 sec
                 frame_skip=125,  # 1 sec
                 goal_threshold=5.,
                 init_altitude=25.,
                 altitude_limits=[11, 75],
                 fire_pos=[-40, 40],
                 fire_dim=[11., 3.5],
                 is_pixels=True):
        super(DroneEnvDiscrete, self).__init__(
            time_limit=time_limit,
            max_no_action_steps=max_no_action_steps,
            frame_skip=frame_skip,
            goal_threshold=goal_threshold,
            init_altitude=init_altitude,
            altitude_limits=altitude_limits,
            fire_pos=fire_pos,
            fire_dim=fire_dim,
            is_pixels=is_pixels)

        # Action space discretized, roll, pitch, and yaw only
        self._limits = np.hstack((self.sim.limits[1, :3],
                                 self.sim.limits[0, :3][::-1]))
        # add 1 for no action
        self.action_space = spaces.Discrete(n=self._limits.shape[-1] + 1)
        self.no_action = self.sim.limits[0] * 0

    def action_map(self, action):
        if action == 0:  # no action
            return self.no_action
        else:
            action -= 1  # reduce the no action
        # Discrete to Box without no action
        encoded = get_one_hot([action], self.action_space.n - 1)
        encoded = self._limits * encoded[0]
        # roll, pitch, yaw and altitud
        decoded = encoded.reshape((2, self.action_space.n // 2))[::-1]  # l, h
        decoded[0] = decoded[0][::-1]  # back order
        decoded = np.hstack((decoded.sum(axis=0), [0]))  # append altitude
        return decoded

    def step(self, action):
        """Do an action step inside the Webots simulator."""
        mapped_action = self.action_map(action)

        return super(DroneEnvDiscrete, self).step(mapped_action)
