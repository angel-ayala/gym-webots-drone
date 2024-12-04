#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:29:06 2024

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import numpy as np

from webots_drone.utils import angle_90deg_offset
from webots_drone.utils import angle_inverse
from webots_drone.utils import compute_distance
from webots_drone.utils import compute_orientation
from webots_drone.utils import compute_risk_distance


class VirtualTarget:
    def __init__(self, dimension=[None, None], webots_node=None, is_3d=False,
                 height_range=[2., 13.], radius_range=[.5, 3.]):
        self.size_range = [height_range, radius_range]
        self.position = [0, 0, 0]
        self.is_3d = is_3d

        # first check if is in a Webots simulation
        self.node = webots_node
        self.set_dimension(*dimension)
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

        self.dimension = [height, radius]
        self.risk_distance = compute_risk_distance(*self.dimension)

        return self.dimension, self.risk_distance

    def get_random_position(self, area):
        height, radius = self.dimension
        random_pos = [0, 0, 0]
        # randomize position
        X_range = [area[0, 0] + radius, area[1, 0] - radius]
        Y_range = [area[0, 1] + radius, area[1, 1] - radius]
        random_pos[0] = self.np_random.uniform(*X_range)
        random_pos[1] = self.np_random.uniform(*Y_range)

        if self.is_3d:
            half_height = height / 2.
            Z_range = [area[0, 2] + half_height, area[1, 2] - half_height]
            random_pos[2] = self.np_random.uniform(*Z_range)

        return np.round(random_pos, 2)

    def set_position(self, position):
        """
        Set the Target Node's position in the scenario.

        :param list pos: The [X, Y, Z] position values where locate the node.
        """
        # ensure fire position inside the flight_area
        target_pos = position

        # set new position
        if self.node is not None:
            target_pos = self.node['get_pos']()
            for i in range(len(position)):
                target_pos[i] = position[i]

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

    def get_orientation(self, reference):
        """Compute the angle between the reference and the target."""
        angle = compute_orientation(reference, self.position)
        angle = angle_inverse(angle_90deg_offset(angle))
        return angle

    def get_height_diff(self, reference):
        """Compute the height difference between the reference and the target."""
        if self.is_3d:
            return (self.position[2] - reference[2]).round(4)
        else:
            return 0.

    def get_elevation_angle(self, reference, norm=False):
        """Compute the levation angle between the reference and the target."""
        # horizontal distance
        h_dist = compute_distance(self.position[:2], reference[:2])
        # vertical difference
        delta_z = self.get_height_diff(reference)
        # elevation angle
        angle = np.arctan2(delta_z, h_dist)
        if norm:
            angle /= np.pi / 2
        return angle

    def get_orientation_diff(self, ref_position, ref_orientation, norm=False):
        """Compute the angle difference between the reference and the target."""
        orientation = self.get_orientation(ref_position)
        diff_angle = ref_orientation - orientation
        if norm:
            diff_angle = np.cos(diff_angle)
        return diff_angle

    def get_risk_distance(self, threshold=0.):
        return self.risk_distance + threshold

    def __repr__(self):
        str_out = f"VirtualTarget(pos=[x: {self.position[0]:.3f}, "
        str_out += f"y: {self.position[1]:.3f}, "
        str_out += f"z: {self.position[2]:.3f}], "
        str_out += f"dim=[h: {self.dimension[0]:.3f}, "
        str_out += f"r: {self.dimension[1]:.3f}])"
        return str_out
