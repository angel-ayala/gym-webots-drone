#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:47:46 2024

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import numpy as np

from webots_drone import WebotsSimulation


class CFSimulation(WebotsSimulation):
    """
    Main class to control the Webots simulation scene.

    In order to work this class, a Robot node must be present in the Webots
    scenario with the supervisor option turned on. For this case the Robot
    node is considered as the RL-agent configured with the Emitter and
    Receiver nodes in order to get and send the states and actions,
    respectively, working as the Remote Control of the drone.
    Additionally, this class is responsible to randomize the fire size and
    location.
    Also, consider the implementation of a default keyboard control as human
    interface for testing purpose.
    """

    def __init__(self):
        super(CFSimulation, self).__init__()
        # replace image size
        self.image_shape = (324, 324, 4)

    @staticmethod
    def get_control_ranges():
        """The control limits to manipulate the angles and altitude."""
        control_ranges = np.array([0.5,     # vel_x (m/s)
                                   0.5,     # vel_y (m/s)
                                   72.,     # yaw_rate (deg/s)
                                   0.5      # vel_z (m/s)
                                   ])
        return np.array([control_ranges * -1,  # low limits
                         control_ranges])      # high limits

    def init_nodes(self):
        """Initialize the target and drone nodes' information."""
        # self.init_areas()
        self.init_target_node()
        self.init_drone_node()

    def take_off(self, height):
        run = True
        vels = [0., 0., 0., self.limits[1][3]]
        drone_alt = 0.0
        while run:
            state = self.get_data()
            if drone_alt < height:
                self.send_action(vels)  # send positive vel_z
                drone_alt += vels[3] * (self.timestep / 1000)
            else:
                self.send_action([0., 0., 0., 0.])  # wait
            run = state['position'][2] < height

    def get_flight_area(self, altitude_limits=[0.25, 2.25]):
        return super().get_flight_area(altitude_limits)


if __name__ == '__main__':
    import traceback
    from webots_simulation import run

    sim_args = {
        'goal_threshold': 0.5,
        'target_pos': [1, -1, 0.5],
        'target_dim': [.025, .02],
        'height_limits': [.25, 2.25],
        'frame_skip': 6,
        'vel_factor': 0.010,
        'is_3d': True,
        'init_height': 0.3
    }

    # run controller
    try:
        controller = CFSimulation()
        run(controller, show=True, **sim_args)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        controller.reset()
