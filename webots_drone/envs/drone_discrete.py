#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:11:52 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import numpy as np
from gym import spaces

from webots_drone.envs import DroneEnvContinuous


class DroneEnvDiscrete(DroneEnvContinuous):
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
                 zone_steps=0):
        super(DroneEnvDiscrete, self).__init__(
            time_limit_seconds=time_limit_seconds,
            max_no_action_seconds=max_no_action_seconds,
            frame_skip=frame_skip,
            goal_threshold=goal_threshold,
            init_altitude=init_altitude,
            altitude_limits=altitude_limits,
            target_pos=target_pos,
            target_dim=target_dim,
            is_pixels=is_pixels,
            zone_steps=zone_steps)

        # Action space discretized, roll, pitch, and yaw only
        control_limits = self.action_limits.copy()
        self._limits = np.hstack((control_limits[1, :3],
                                  control_limits[0, :3][::-1]))
        # add 1 for no action
        self.action_space = spaces.Discrete(n=self._limits.shape[-1] + 1)

    def discrete2continuous(self, action):
        if action == 0:  # no action
            return self.action_limits[1] * 0.
        else:
            limits = self.action_limits[:, :3]
            mask = np.eye(limits.shape[-1])
            action_map = [limits[a % 2 - 1] * mask[a // 2]
                          for a in range(np.multiply(*limits.shape))]
            # get action and append altitude
            decoded = np.hstack((action_map[action - 1], [0]))
            return decoded

    def step(self, action):
        """Do an action step inside the Webots simulator."""
        mapped_action = self.discrete2continuous(action)

        return super(DroneEnvDiscrete, self).step(mapped_action)
