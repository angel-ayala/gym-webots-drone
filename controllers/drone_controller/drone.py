#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:49:45 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import numpy as np

from webots_drone.utils import bytes2image


class Drone:
    """The Drone class manage each sensor and actuators of the drone.

    It is developed for the Mavic 2 Pro drone, it consists of GPS, IMU, Gyro,
    Compass, Camera, LED and Motor nodes.
    This drone control unit is designed to stabilize the drone through 4 PID
    controllers tunned for a 8ms simulation timestep, and the drone's gimbal
    with a Damping node in the WorldInfo node with values of 0.5 for both
    angular and linear fields.

    :param integer timestep: The simulation timestep, 8ms mus be setted,
        unexpected behaviour can occur with a different value.
    :param string name: A name for the controller, just for debug purpose.
    :param float start_alt: The initial altitude to be reached.
    """

    def __init__(self, robot):
        # Time helpers
        self.time_counter = 0

        # Variables
        self.lift_thrust = 68.5  # with this thrust, the drone lifts.
        self.init_dist_sensors(robot, int(robot.getBasicTimeStep()))
        self.init_devices(robot, int(robot.getBasicTimeStep()))
        self._position = np.array([0.0, 0.0, 0.0])

    def init_dist_sensors(self, drone_node, timestep):
        """Initialize each sensor distance of the Mavic 2 Pro.

        :param drone_node Robot: The instantiated Robot Node class.
        :param integer timestep: The simulation timestep, 8ms mus be setted,
            unexpected behaviour can occur with a different value.
        """
        self.sensors_id = ['front left dist sonar',
                           'front right dist sonar',
                           'rear top dist sonar',
                           'rear bottom dist sonar',
                           'left side dist sonar',
                           'right side dist sonar',
                           'down front dist sonar',
                           'down back dist sonar',
                           'top dist infrared']
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
        self.imu = drone_node.getDevice("inertial unit")
        self.imu.enable(timestep)
        # Acceleration angles [roll, pitch, yaw]
        self.gyro = drone_node.getDevice("gyro")
        self.gyro.enable(timestep)
        # Direction degree with north as reference
        self.compass = drone_node.getDevice("compass")
        self.compass.enable(timestep)

        # Video acquisition
        fps = 25
        self.camera = drone_node.getDevice("camera")
        self.camera_rate = 1000 // fps
        self.camera.enable(self.camera_rate)

        # LEDS
        self.leds = [
            drone_node.getDevice("front left led"),
            drone_node.getDevice("front right led")
        ]

        # Gimbal
        self.camera_roll = drone_node.getDevice("camera roll")
        self.camera_pitch = drone_node.getDevice("camera pitch")

        # Motors
        self.motors_id = ['front left propeller',
                          'front right propeller',
                          'rear left propeller',
                          'rear right propeller']
        self.motors = list()
        for mid in self.motors_id:
            motor = drone_node.getDevice(mid)
            motor.setPosition(float('inf'))
            motor.setVelocity(1.)
            self.motors.append(motor)

        return True

    def blink_leds(self):
        """Blink the LED nodes."""
        led_state = int(self.time_counter) % 2
        self.leds[0].set(led_state)
        self.leds[1].set(int(not led_state))

    def gimbal_stabilize(self):
        """Stabilize camera (gimbal)."""
        acceleration = self.gyro.getValues()
        self.camera_roll.setPosition(-0.115 * acceleration[0])
        self.camera_pitch.setPosition(-0.1 * acceleration[1])

    def get_odometry(self):
        """Get the drone's current acceleration, angles and position."""
        orientation = self.imu.getRollPitchYaw()
        angular_velocity = self.gyro.getValues()
        position = self.gps.getValues()
        speed = self.gps.getSpeedVector()
        compass = self.compass.getValues()
        north_rad = np.arctan2(compass[1], compass[0])

        return orientation, angular_velocity, position, speed, north_rad

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

    def set_motors_velocity(self, fl_motor, fr_motor, rl_motor, rr_motor):
        """Set the drone's motor velocity."""
        # Actuate over the motors
        if not np.isnan(fl_motor):
            self.motors[0].setVelocity(self.lift_thrust + fl_motor)
            self.motors[1].setVelocity(-(self.lift_thrust + fr_motor))
            self.motors[2].setVelocity(-(self.lift_thrust + rl_motor))
            self.motors[3].setVelocity(self.lift_thrust + rr_motor)
