#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:29:06 2024

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import numpy as np

from webots_drone.utils import compute_distance
from webots_drone.utils import compute_risk_distance


class VirtualTarget:
    def __init__(self, position=None, dimension=None, webots_node=None,
                 default_height=[2., 13.], default_radius=[.5, 3.],
                 is_3d=False):
        self.default_size = [default_height, default_radius]

        # first check if is in a Webots simulation
        self.node = None
        if webots_node is not None:
            self.node = webots_node

        if position is None:
            self.position = []
        if dimension is None:
            self.dimension = []

        self.is_3d = is_3d

        self.seed()

    def seed(self, seed=None):
        """Set seed for the numpy.random, None default."""
        self.np_random = np.random.RandomState(seed)
        # world_node = self.getFromDef('World')
        # world_node.getField('randomSeed').setSFInt32(
        #     0 if seed is None else seed)
        return seed

    def set_dimension(self, height=None, radius=None):
        """
        Set the Target Node's height and radius.

        :param float height: The fire's height, default is 2.
        :param float radius: The fire's radius, default is 0.5

        :return float, float: the settled height and radius values.
        """
        if height is None:
            height = self.np_random.uniform(*self.default_size[0])
        if radius is None:
            radius = self.np_random.uniform(*self.default_size[1])

        # node fields
        if self.node is not None:
            self.node['set_height'](float(height))
            self.node['set_radius'](float(radius))

            # correct position in Z axis and update risk_distance value
            target_pos = self.node['get_pos']()
            target_pos[2] -= height * 0.5  # update height
            self.node['set_pos'](list(target_pos))
            self.position = target_pos

        self.dimension = (height, radius)
        self.risk_distance = compute_risk_distance(*self.dimension)

        return self.dimension, self.risk_distance

    def set_position(self, flight_area, position=None):
        """
        Set the Target Node's position in the scenario.

        Set a desired node position value or generated a new random one inside
        the scenario's forest area if no input is given.

        :param list flight_area: The [X, Y, Z] lower and upper values of the 
            available flight area.
        :param list pos: The [X, Y, Z] position values where locate the node,
            if no values are given a random one is generated instead.
            Default is None.
        """
        if self.node is not None:
            radius = self.node['get_radius']()
            target_pos = self.node['get_pos']()  # current position
        else:
            radius = self.dimension[1]
            target_pos = self.position

        # get forest limits
        X_range = flight_area[:, 0]
        Y_range = flight_area[:, 1]
        Z_range = flight_area[:, 2]

        if position is None:
            # randomize position
            target_pos[0] = self.np_random.uniform(radius - abs(X_range[0]),
                                                   X_range[1] - radius)
            target_pos[1] = self.np_random.uniform(radius - abs(Y_range[0]),
                                                   Y_range[1] - radius)
            if self.is_3d:
                target_pos[2] = self.np_random.uniform(*Z_range)
        else:
            target_pos[0] = position[0]
            target_pos[1] = position[1]
            if self.is_3d:
                target_pos[2] = position[2]

        # ensure fire position inside the forest
        target_pos = np.clip(target_pos, *flight_area)

        # set new position
        if self.node is not None:
            self.node['set_pos'](list(target_pos))
        self.position = target_pos

        return self.position

    def get_distance(self, reference):
        """Compute the distance between the reference and the target."""
        target_pos = self.position
        # consider only xy coordinates
        if not self.is_3d:
            target_pos[2] = reference[2]
        # Squared Euclidean distance
        return compute_distance(reference, target_pos)

    def get_risk_distance(self, threshold=0.):
        return self.risk_distance + threshold
