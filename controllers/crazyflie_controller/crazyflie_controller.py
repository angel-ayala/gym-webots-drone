#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:45:07 2024

@author: angel
"""

import numpy as np
from controller import Robot

from crazyflie_quadcopter import CFQuadcopter
from webots_drone.utils import angle_90deg_offset
from webots_drone.utils import angle_inverse
from webots_drone.utils import encode_image
from webots_drone.utils import emitter_send_json
from webots_drone.utils import receiver_get_json


import sys
# Change this path to your crazyflie-firmware folder
sys.path.append('../../../crazyflie-firmware')
import cffirmware


# Drone Robot
class CFController(Robot):
    """CFController is the main class to manage the Crayflie drone's action.

    This class manage the Emitter and Receiver nodes to send and get the states
    and actions of the drone to the Remote Control.
    This controls the motors velocity and sensors of the drone.
    """

    def __init__(self):
        super(CFController, self).__init__()
        # local variables
        self.timestep = int(self.getBasicTimeStep())

        # Initialize Flight Control
        print('Initializing Crazyflie Control...', end=' ')
        self.__drone = CFQuadcopter(self)
        self.__time_delta = self.timestep / 1000
        self.__motors_controller()

        # Initialize comms
        self.state = self.getDevice('StateEmitter')  # channel 4
        self.action = self.getDevice('ActionReceiver')  # channel 6
        self.action.enable(self.timestep)
        print('OK')

    def __motors_controller(self):
        # Initialize Firmware PID controller
        cffirmware.controllerPidInit()
        self.fix_height = 0.0

    def __compute_disturbances(self, disturbances):
        # current state
        orientation, ang_velocity, position, speed = self.__drone.get_odometry()

        # firmware estimations update
        # TODO replace these with a EKF python binding
        state = cffirmware.state_t()
        state.attitude.roll = np.degrees(orientation[0])
        state.attitude.pitch = -np.degrees(orientation[1])
        state.attitude.yaw = np.degrees(orientation[2])
        state.position.x = position[0]
        state.position.y = position[1]
        state.position.z = position[2]
        state.velocity.x = speed[0]
        state.velocity.y = speed[1]
        state.velocity.z = speed[2]

        # Put gyro in sensor data
        sensors = cffirmware.sensorData_t()
        sensors.gyro.x = np.degrees(ang_velocity[0])
        sensors.gyro.y = np.degrees(ang_velocity[1])
        sensors.gyro.z = np.degrees(ang_velocity[2])

        ## Fill in Setpoints
        setpoint = cffirmware.setpoint_t()
        setpoint.mode.x = cffirmware.modeVelocity
        setpoint.mode.y = cffirmware.modeVelocity
        setpoint.mode.yaw = cffirmware.modeVelocity
        setpoint.mode.z = cffirmware.modeAbs

        setpoint.velocity.x = disturbances[0]
        setpoint.velocity.y = disturbances[1]
        setpoint.attitudeRate.yaw = np.degrees(disturbances[2])

        self.fix_height += disturbances[3] * self.__time_delta
        setpoint.position.z = self.fix_height

        setpoint.velocity_body = True

        # Firmware PID bindings
        control = cffirmware.control_t()
        tick = 100  # this value makes sure that the position controller and attitude controller are always always initiated
        cffirmware.controllerPid(control, setpoint, sensors, state, tick)

        # Degrees to radians
        cmd_roll = np.radians(control.roll)
        cmd_pitch = np.radians(control.pitch)
        cmd_yaw = -np.radians(control.yaw)
        cmd_thrust = control.thrust

        # return motor velocities
        pose_disturbance = [cmd_roll,
                            cmd_pitch,
                            cmd_yaw,
                            cmd_thrust]
        return pose_disturbance

    def __compute_velocity(self):
        # compute disturbances velocities
        command, _ = receiver_get_json(self.action)
        disturbances = (command['disturbances']
                        if len(command.keys()) > 0 else [0., 0., 0., 0.])
        # apply disturbances velocities
        pose_disturbance = self.__compute_disturbances(disturbances)
        cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust = pose_disturbance
        motorPower_m1 = cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
        motorPower_m2 = cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m3 = cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m4 = cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw

        return motorPower_m1, motorPower_m2, motorPower_m3, motorPower_m4

    def __send_state(self):
        # get current state
        uav_orientation, uav_angular_velocity, \
            uav_position, uav_speed = self.__drone.get_odometry()
        uav_north_rad = uav_orientation[2]
        uav_distance_sensors = self.__drone.get_dist_sensors()
        uav_image = self.__drone.get_image()

        # Compass from yaw angle in ENU reference system
        uav_north_rad = angle_inverse(angle_90deg_offset(uav_north_rad))

        # read motors current velocity
        motors_vel = [m.getVelocity() for m in self.__drone.motors]

        # encode data
        msg_data = dict(timestamp=np.round(self.getTime(), 3),
                        orientation=uav_orientation,
                        angular_velocity=uav_angular_velocity,
                        position=uav_position,
                        speed=uav_speed,
                        north=uav_north_rad,
                        dist_sensors=uav_distance_sensors,
                        motors_vel=motors_vel)
        enc_img = "NoImage" if uav_image is None else encode_image(uav_image)
        msg_data['image'] = enc_img
        # send data
        emitter_send_json(self.state, msg_data)

    def run(self):
        """Run controller's main loop.

        Send the variations of the altitude, the roll, pitch, and yaw angles
        to the drone. Send the current image captured by the drone's
        camera and get the actions from the Remote Control, once the action
        (variations of the angles an altitude) is received, the Drone
        calculates the velocity, and apply it to the 3 different angles and
        altitude.
        """
        # control loop
        print('Drone control is active')
        while self.step(self.timestep) != -1:
            # actuates over devices and motors
            propellers_vel = self.__compute_velocity()
            self.__drone.set_motors_velocity(*propellers_vel)
            # comms
            self.__send_state()


if __name__ == '__main__':
    # run controller
    controller = CFController()
    controller.run()
    del controller
