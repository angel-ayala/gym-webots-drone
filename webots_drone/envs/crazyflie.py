#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:40:27 2024

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import numpy as np
from gym import logger
from gym import spaces

from webots_drone.envs import DroneEnvContinuous


class CrazyflieEnvContinuous(DroneEnvContinuous):
    """Gym enviroment to control the Crazyflie drone in Webots simulator."""

    def __init__(self, time_limit_seconds=60,  # 1 min
                 max_no_action_seconds=5,  # 5 sec
                 frame_skip=6,  # 1 sec
                 goal_threshold=0.5,
                 init_altitude=0.3,
                 altitude_limits=[0.25, 2.25],
                 fire_pos=2,
                 fire_dim=[.05, .02],
                 is_pixels=False,
                 zone_steps=10):
        super(CrazyflieEnvContinuous, self).__init__(
            time_limit_seconds=time_limit_seconds,  # 1 min
            max_no_action_seconds=max_no_action_seconds,  # 5 sec
            frame_skip=frame_skip,  # 1 sec
            goal_threshold=goal_threshold,
            init_altitude=init_altitude,
            altitude_limits=altitude_limits,
            fire_pos=fire_pos,
            fire_dim=fire_dim,
            is_pixels=is_pixels,
            zone_steps=zone_steps)

    def init_sim(self):
        # Simulation controller
        from webots_drone import CFSimulation
        logger.info('Checking Webots connection...')
        self.sim = CFSimulation()
        logger.info('Connected to Webots')

    def set_reaction_intervals(self, frame_skip):
        self._frame_inter = [frame_skip - 1, frame_skip + 1]

    def create_quadrants(self):
        # offset the flight_area
        target_area = self.flight_area.copy()
        # adjust x- y-axis
        target_area[0, :2] += 0.2
        target_area[1, :2] -= 0.2
        # adjust z-axis
        target_area[0, 2] += 0.25
        target_area[1, 2] -= 0.45
        # compute quadrants coordinates
        area = np.linspace(target_area[0], target_area[1], 3)
        area_points = []
        for z in area[:, 2]:
            for x in area[:, 0]:
                for y in area[:, 1]:
                    if x == y == 0:
                        continue
                    area_points.append([x, y, z])
        area_points = np.asarray(area_points)

        # Function to order points clockwise around the origin in the x-y plane
        def clockwise_sort(points):
            # Calculate the angle of each point relative to the center (0, 0)
            angles = np.arctan2(points[:, 1], points[:, 0])
            # Sort by angle in descending order for clockwise
            return points[np.argsort(-angles)]

        # Separate area_points by unique z levels
        sorted_area_points = []
        for z in np.unique(area_points[:, 2]):
            # Get only (x, y) for sorting
            level_points = area_points[area_points[:, 2] == z][:, :2]
            sorted_level_points = clockwise_sort(level_points)
            # Append z back to each point and add to sorted list
            sorted_area_points.extend(
                [np.append(p, z) for p in sorted_level_points])

        # Convert back to np.array for easier handling
        return np.asarray(sorted_area_points)

    def compute_reward(self, obs, info, is_3d=True, vel_factor=0.02,
                       pos_thr=0.0001):
        return super().compute_reward(obs, info, is_3d, vel_factor, pos_thr)

    def create_target(self, position=None, dimension=None):
        # virtualTarget
        vtarget = super().create_target(position, dimension)
        vtarget.is_3d = True
        return vtarget

    def lift_uav(self):
        self.sim.take_off(self.init_altitude)

    def no_action_limit(self, position, pos_thr=0.0001):
        return super().no_action_limit(position, pos_thr)


class CrazyflieEnvDiscrete(CrazyflieEnvContinuous):
    """Gym enviroment to control the Crazyflie drone in Webots simulator."""

    def __init__(self, time_limit_seconds=60,  # 1 min
                 max_no_action_seconds=5,  # 5 sec
                 frame_skip=6,  # 1 sec
                 goal_threshold=0.5,
                 init_altitude=0.3,
                 altitude_limits=[0.25, 2.25],
                 fire_pos=2,
                 fire_dim=[.025, .02],
                 is_pixels=False,
                 zone_steps=10):
        super(CrazyflieEnvDiscrete, self).__init__(
            time_limit_seconds=time_limit_seconds,  # 1 min
            max_no_action_seconds=max_no_action_seconds,  # 5 sec
            frame_skip=frame_skip,  # 1 sec
            goal_threshold=goal_threshold,
            init_altitude=init_altitude,
            altitude_limits=altitude_limits,
            fire_pos=fire_pos,
            fire_dim=fire_dim,
            is_pixels=is_pixels,
            zone_steps=zone_steps)

        # Action space discretized + no action
        control_limits = np.hstack((self.action_limits[1],
                                    self.action_limits[0][::-1]))
        self.action_space = spaces.Discrete(n=control_limits.shape[-1] + 1)

    def discrete2continuous(self, action):
        if action == 0:  # no action
            return self.action_limits[1] * 0.
        else:
            limits = self.action_limits.copy()
            mask = np.eye(limits.shape[-1])
            action_map = [limits[a % 2 - 1] * mask[a // 2]
                          for a in range(np.multiply(*limits.shape))]
            decoded = action_map[action - 1]
            return decoded

    def step(self, action):
        """Do an action step inside the Webots simulator."""
        mapped_action = self.discrete2continuous(action)

        return super(CrazyflieEnvDiscrete, self).step(mapped_action)
