#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:35:22 2024

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import numpy as np
from webots_drone.utils import bytes2image


class CFQuadcopter:
    """The CFQuadcopter class manage each sensor and actuators of the drone.

    It is developed for the Crazyflie 2.1 nano drone, it consists of IMU, GPS,
    Gyro, Camera, Range sensors and Motor nodes.
    """

    def __init__(self, robot):
        # Time helpers
        self.time_counter = 0

        # Variables
        self.init_dist_sensors(robot, int(robot.getBasicTimeStep()))
        self.init_devices(robot, int(robot.getBasicTimeStep()))
        self._position = np.array([0.0, 0.0, 0.0])

    def init_dist_sensors(self, drone_node, timestep):
        """Initialize each sensor distance of the Mavic 2 Pro.

        :param drone_node Robot: The instantiated Robot Node class.
        :param integer timestep: The simulation timestep, 8ms mus be setted,
            unexpected behaviour can occur with a different value.
        """
        self.sensors_id = ['range_front',
                           'range_left',
                           'range_back',
                           'range_right',]
        # instantiate distance sensors
        self.sensors = list()
        for sid in self.sensors_id:
            sensor = drone_node.getDevice(sid)
            sensor.enable(timestep)
            self.sensors.append(sensor)

        return True

    def init_devices(self, drone_node, timestep):
        """Initialize each device of the Mavic 2 Pro, in a desired timestep.

        The camera node is initialized at 33ms timestep to reach ~30fps.

        :param drone Robot: The instantiated Robot Node class.
        :param integer timestep: The simulation timestep, 8ms mus be setted,
            unexpected behaviour can occur with a different value.
        """
        # Position coordinates [X, Y, Z]
        self.gps = drone_node.getDevice("gps")
        self.gps.enable(timestep)
        # Angles respect global coordinates [roll, pitch, yaw]
        self.imu = drone_node.getDevice("inertial_unit")
        self.imu.enable(timestep)
        # Acceleration angles [roll, pitch, yaw]
        self.gyro = drone_node.getDevice("gyro")
        self.gyro.enable(timestep)

        # Video acquisition
        fps = 25
        self.camera = drone_node.getDevice("camera")
        self.camera_rate = 1000 // fps
        self.camera.enable(self.camera_rate)

        # Motors
        self.motors_id = ['m1_motor',
                          'm2_motor',
                          'm3_motor',
                          'm4_motor']
        self.motors = list()
        for mid in self.motors_id:
            motor = drone_node.getDevice(mid)
            motor.setPosition(float('inf'))
            if mid in ['m1_motor', 'm3_motor']:
                motor.setVelocity(-1.)
            else:
                motor.setVelocity(1.)
            self.motors.append(motor)

        return True

    def get_odometry(self):
        """Get the drone's current acceleration, angles and position."""
        orientation = self.imu.getRollPitchYaw()
        angular_velocity = self.gyro.getValues()
        position = self.gps.getValues()
        speed = self.gps.getSpeedVector()

        return orientation, angular_velocity, position, speed

    def get_image(self):
        """Get the Camera node image with size and channels.

        :return the data image with BGRA values
        """
        camera_image = None
        if self.camera.getImage():
            camera_image = bytes2image(self.camera.getImage(),
                                       self.get_camera_image_shape())
        return camera_image

    def get_dist_sensors(self):
        """Get the Distance sensors Nodes' measurements."""
        sensors = dict()
        for i, sensor_name in enumerate(self.sensors_id):
            dist_sensor = self.sensors[i]
            sensors[sensor_name] = [dist_sensor.getValue(),
                                    dist_sensor.getMinValue(),
                                    dist_sensor.getMaxValue()]
        return sensors

    def get_camera_image_shape(self):
        """Get the camera image dimension and channels."""
        return (self.camera.getHeight(), self.camera.getWidth(), 4)  # channels

    def set_motors_velocity(self, m1_power, m2_power, m3_power, m4_power,
                            scaling=1000):
        """Set the drone's motor velocity."""
        # Actuate over the motors
        if not np.isnan(m1_power):
            self.motors[0].setVelocity(-m1_power/scaling)
            self.motors[1].setVelocity(m2_power/scaling)
            self.motors[2].setVelocity(-m3_power/scaling)
            self.motors[3].setVelocity(m4_power/scaling)
