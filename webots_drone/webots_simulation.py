#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:19:44 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import os
import sys
sys.path.append(os.environ['WEBOTS_HOME'] + "/lib/controller/python")
import traceback
import numpy as np
from controller import Supervisor

from webots_drone.utils import receiver_get_json
from webots_drone.utils import emitter_send_json
from webots_drone.utils import decode_image
from webots_drone.utils import min_max_norm
from webots_drone.utils import compute_distance
from webots_drone.utils import compute_flight_area


# Webots environment controller
class WebotsSimulation(Supervisor):
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
        super(WebotsSimulation, self).__init__()
        # simulation timestep
        self.timestep = int(self.getBasicTimeStep())
        self.image_shape = (240, 400, 4)
        self._data = dict()
        # actions value boundaries
        self.set_limits()
        # runtime vars
        # self.seed()
        self.init_nodes()
        self.init_comms()

    @property
    def is_running(self):
        """Get if the simulation is running."""
        return self.SIMULATION_MODE_PAUSE != self.simulationGetMode()

    def pause(self):
        """Pause the Webots's simulation."""
        self.simulationSetMode(self.SIMULATION_MODE_PAUSE)

    def play(self):
        """Start the Webots's simulation in real time mode."""
        self.simulationSetMode(self.SIMULATION_MODE_REAL_TIME)

    def play_fast(self):
        """Start the Webots's simulation in fast mode."""
        self.simulationSetMode(self.SIMULATION_MODE_FAST)

    def seed(self, seed=None):
        """Set seed for the numpy.random and WorldInfo node, None default."""
        self.np_random = np.random.RandomState(seed)
        # world_node = self.getFromDef('World')
        # world_node.getField('randomSeed').setSFInt32(
        #     0 if seed is None else seed)
        return seed

    def set_limits(self):
        """Get the limits to manipulate the angles and altitude."""
        # limits = np.array([np.pi / 12.,      # roll
        #                    np.pi / 12.,      # pitch
        #                    np.pi / 360.,     # yaw
        #                    8.               # altitude
        #                    ])
        limits = np.array([2.,  # roll
                           2.,  # pitch
                           2.,  # yaw
                           8.   # altitude
                           ])
        self.limits = np.array([limits * -1,  # low limist
                                limits])      # high limits
        return self
    
    def get_flight_area(self, altitude_limits):
        # rc_pos = self.getSelf().getPosition()
        # offset = [[50, 50, altitude_limits[0]],
        #           [50, 40, altitude_limits[1]]]
        # return compute_flight_area(rc_pos, offset)
        area_size = self.getFromDef('FlightArea').getField('size').getSFVec2f()
        area_size = [fs / 2 for fs in area_size]  # size from center
        flight_area = [[fs * -1 for fs in area_size], area_size]
        flight_area[0].append(altitude_limits[0])
        flight_area[1].append(altitude_limits[1])
        return flight_area

    def init_comms(self):
        """Initialize the communication nodes."""
        self.action = self.getDevice('ActionEmitter')  # channel 6
        self.state = self.getDevice('StateReceiver')  # channel 4
        self.state.enable(self.timestep)
        return self

    def init_areas(self):
        # Flight Area
        # area_size = self.getFromDef('FlightArea').getField('size').getSFVec2f()
        # area_size = [fs / 2 for fs in area_size]  # size from center
        # self.flight_area = [[fs * -1 for fs in area_size], area_size]

        # Forest area
        forest_shape = self.getFromDef('ForestArea').getField('shape')
        self.forest_area = []

        for i in range(forest_shape.getCount()):
            self.forest_area.append(forest_shape.getMFVec2f(i))
        return self.forest_area

    def init_target_node(self):
        # Fire vars
        target_node = self.getFromDef('FireSmoke')
        self.target_node = dict(
            node=target_node,
            get_height=lambda: target_node.getField('fireHeight').getSFFloat(),
            get_radius=lambda: target_node.getField('fireRadius').getSFFloat(),
            get_pos=lambda: np.array(
                target_node.getField('translation').getSFVec3f()),
            set_height=target_node.getField('fireHeight').setSFFloat,
            set_radius=target_node.getField('fireRadius').setSFFloat,
            set_pos=target_node.getField('translation').setSFVec3f
            )
        self.risk_distance = self.target_node['get_radius']() +\
            self.target_node['get_height']() * 4

    def init_drone_node(self):
        # Drone vars
        drone_node = self.getFromDef('Drone')
        self.drone_node = dict(
            node=drone_node,
            get_pos=lambda: np.array(
                drone_node.getField('translation').getSFVec3f())
            )

    def init_nodes(self):
        """Initialize the target and drone nodes' information."""
        self.init_areas()
        self.init_target_node()
        self.init_drone_node()

    def reset(self):
        """Reset the Webots simulation.

        Set the simulation mode with the constant SIMULATION_MODE_PAUSE as
        defined in the Webots documentation.
        Reset the fire and drone nodes at the starting point, and restart
        the controllers simulation.
        """
        if self.is_running:
            self.state.disable()  # prevent to receive data
            self.target_node['node'].restartController()
            self.drone_node['node'].restartController()
            self.simulationReset()
            self.simulationResetPhysics()
            # stop simulation
            self.one_step()  # step to process the reset
            self.pause()
            self.state.enable(self.timestep)
            self._data = dict()

    def one_step(self):
        """Do a Robot.step(timestep)."""
        self.step(self.timestep)

    def set_fire_dim(self, fire_height=7., fire_radius=5.):
        """
        Set the FireSmoke Node's height and radius.

        :param float fire_height: The fire's height, default is 2.
        :param float fire_radius: The fire's radius, default is 0.5

        :return float, float: the settled height and radius values.
        """
        # FireSmoke node fields
        self.target_node['set_height'](fire_height)
        self.target_node['set_radius'](fire_radius)

        # update position and risk_zone
        self.set_fire_position(fire_pos=self.target_node['get_pos']())

        return fire_height, fire_radius

    def set_fire_position(self, fire_pos=None):
        """
        Set the FireSmoke Node's position in the scenario.

        Set a desired node position value or generated a new random one inside
        the scenario's forest area if no input is given.

        :param list pos: The [X, Z] position values where locate the node, if
            no values are given a random one is generated instead.
            Default is None.
        """
        fire_radius = self.target_node['get_radius']()
        # print(self.forest_area)
        if fire_pos is None:  # randomize position
            fire_p = self.target_node['get_pos']()  # current position
            # get forest limits
            X_range = [self.forest_area[3][0], self.forest_area[1][0]]
            Y_range = [self.forest_area[1][1], self.forest_area[3][1]]

            # randomize position
            fire_p[0] = self.np_random.uniform(fire_radius - abs(X_range[0]),
                                               X_range[1] - fire_radius)
            fire_p[1] = self.np_random.uniform(fire_radius - abs(Y_range[0]),
                                               Y_range[1] - fire_radius)
        else:
            fire_p = [fire_pos[0], fire_pos[1], 0]

        fire_height = self.target_node['get_height']()
        fire_p[2] = fire_height * 0.5  # update height
        # print('fire_pos', fire_p)

        # FireSmoke node fields
        self.target_node['set_pos'](list(fire_p))
        self.risk_distance = fire_radius + fire_height * 4

        return fire_p, self.risk_distance

    def randomize_fire_position(self):
        """Randomize the size and position of the FireSmoke node.

        The size and position has value in meters.
        The height of the node is [2., 13.] and it radius is [0.5, 3.]
        The position is directly related to the radius, reducing the 2-axis
        available space and requiring move up given its height.
        The position is delimited by the forest area.
        """
        # randomize dimension
        fire_height, fire_radius = self.set_fire_dim(
            fire_height=self.np_random.uniform(2., 13.),
            fire_radius=self.np_random.uniform(0.5, 3.))

        # avoid to the fire appears near the drone's initial position
        n_random = 0
        while (self.get_target_distance() <= self.risk_distance
               or n_random == 0):
            # randomize position
            fire_pos, self.risk_distance = self.set_fire_position()
            n_random += 1

        return fire_pos, fire_height, fire_radius, self.risk_distance

    def get_drone_pos(self):
        """Read the current drone position from the node's info."""
        return self.drone_node['get_pos']()

    def get_target_pos(self):
        """Read the current target position from the node's info."""
        return self.target_node['get_pos']()

    def get_target_distance(self):
        """Compute the drone's distance to the fire."""
        fire_position = self.get_target_pos()
        drone_position = self.get_drone_pos()
        # consider only xy coordinates
        fire_position[2] = drone_position[2]
        # Squared Euclidean distance
        distance = compute_distance(drone_position, fire_position)
        return distance

    def read_data(self):
        """Read the data sended by the drone's Emitter node.

        Capture and translate the drones sended data with the Receiver node.
        This data is interpreted as the drone's state
        """
        # capture UAV sensors
        uav_state, emitter_info = receiver_get_json(self.state)

        if len(uav_state.keys()) == 0:
            return self._data

        timestamp = uav_state['timestamp']
        orientation = uav_state['orientation']
        angular_velocity = uav_state['angular_velocity']
        position = uav_state['position']
        speed = uav_state['speed']
        north_deg = uav_state['north']
        dist_sensors = list()

        # Normalize angular values
        orientation[0] = min_max_norm(orientation[0],  # roll
                                      a=-1, b=1,
                                      minx=-np.pi, maxx=np.pi)
        orientation[1] = min_max_norm(orientation[1],  # pitch
                                      a=-1, b=1,
                                      minx=-np.pi/2, maxx=np.pi/2)
        orientation[2] = min_max_norm(orientation[2],  # yaw
                                      a=-1, b=1,
                                      minx=-np.pi, maxx=np.pi)

        # Normalize distance sensor values
        for idx, sensor in uav_state['dist_sensors'].items():
            if sensor[2] == sensor[1] == sensor[0] == 0.:
                continue
            s_val = min_max_norm(sensor[0],
                                 a=0, b=1,
                                 minx=sensor[1], maxx=sensor[2])
            dist_sensors.append(s_val)

        # Normalize north degree
        north_deg = min_max_norm(north_deg,
                                  a=-1, b=1,
                                  minx=0, maxx=360)

        if type(uav_state['image']) == str and uav_state['image'] == "NoImage":
            img = np.zeros(self.image_shape)
        else:
            img = decode_image(uav_state['image'])

        self._data = dict(timestamp=timestamp,
                          orientation=orientation,
                          angular_velocity=angular_velocity,
                          position=position,
                          speed=speed,
                          north_deg=north_deg,
                          dist_sensors=dist_sensors,
                          motors_vel=uav_state['motors_vel'],
                          image=img,
                          emitter=emitter_info,
                          rc_position=self.getSelf().getPosition(),
                          target_position=self.target_node['get_pos']())

    def get_data(self):
        return self._data.copy()

    def send_data(self, data):
        # send data and do a Robot.step
        command = data
        command['timestamp'] = self.getTime()
        emitter_send_json(self.action, command)
        self.one_step()  # step to process the action
        self.read_data()

    def sync(self):
        # sync data
        while len(self._data.keys()) == 0:
            self.one_step()
            self.read_data()
        # sync orientation
        command = dict(disturbances=self.limits[1].tolist())
        self.send_data(command)

    def __del__(self):
        """Stop simulation when is destroyed."""
        try:
            self.reset()
        except Exception as e:
            print('ERROR: unable to reset the environment!')
            traceback.print_tb(e.__traceback__)
            print(e)


if __name__ == '__main__':
    import cv2

    def print_control_keys():
        """Display manual control message."""
        print("You can control the drone with your computer keyboard:")
        print("IMPORTANT! The Webots 3D window must be selected to work!")
        print("- 'up': move forward.")
        print("- 'down': move backward.")
        print("- 'right': strafe right.")
        print("- 'left': strafe left.")
        print("- 'w': increase the target altitude.")
        print("- 's': decrease the target altitude.")
        print("- 'd': turn right.")
        print("- 'a': turn left.")
        print("- 'q': exit.")

    def run(controller, show=True):
        """Run controller's main loop.

        Capture the keyboard and translate into fixed float values to variate
        the 3 different angles and the altitude, optionally an image captured
        from the drone's camera can be presented in a new window.
        The pitch and roll angles are variated in +-pi/12.,
        the yaw angle in +-pi/360. and the altitude in +-5cm.
        The control keys are:
            - ArrowUp:      +pitch
            - ArrowDown:    -pitch
            - ArrowLeft:    -roll
            - ArrowRight:   +roll
            - W:            +altitude
            - S:            -altitude
            - A:            +yaw
            - D:            +yaw
            - Q:            EXIT

        :param bool show: Set if show or not the image from the drone's camera.
        """
        # keyboard interaction
        print_control_keys()
        kb = controller.getKeyboard()
        kb.enable(controller.timestep)

        # Start simulation with random FireSmoke position
        controller.seed()
        controller.randomize_fire_position()
        controller.play()
        controller.sync()
        run_flag = True

        print('Fire scene is running')
        while (run_flag):  # and drone.getTime() < 30):
            # capture control data
            key = kb.getKey()

            roll_angle = 0.
            pitch_angle = 0.
            yaw_angle = 0.  # drone.yaw_orientation
            altitude = 0.  # drone.target_altitude

            while key > 0:
                # roll
                if key == kb.LEFT:
                    roll_angle = controller.limits[0][0]
                elif key == kb.RIGHT:
                    roll_angle = controller.limits[1][0]
                # pitch
                elif key == kb.UP:
                    pitch_angle = controller.limits[0][1]
                elif key == kb.DOWN:
                    pitch_angle = controller.limits[1][1]
                # yaw
                elif key == ord('D'):
                    yaw_angle = controller.limits[0][2]
                elif key == ord('A'):
                    yaw_angle = controller.limits[1][2]
                # altitude
                elif key == ord('S'):
                    altitude = controller.limits[0][3]  # * 0.1
                elif key == ord('W'):
                    altitude = controller.limits[1][3]  # * 0.1
                # quit
                elif key == ord('Q'):
                    print('Terminated')
                    run_flag = False
                key = kb.getKey()

            action = dict(disturbances=[
                roll_angle,
                pitch_angle,
                yaw_angle,
                altitude
            ])
            # perform action
            controller.send_data(action)

            # capture state
            state_data = controller.get_data()
            if show and state_data is not None:
                cv2.imshow("Drone's live view", state_data['image'])
                cv2.waitKey(1)
            # print("DIST: {:.2f} [{:.2f}]".format(
            #     controller.get_goal_distance(), controller.risk_distance),
            #     "(INFO:",
            #     # "obj_det: {},".format(
            #     # controller.check_near_object(sensors)),
            #     # "out_alt: {},".format(
            #     # controller.check_altitude()),
            #     # "out_area: {},".format(
            #     # controller.check_flight_area()),
            #     # "is_flip: {},".format(
            #     # controller.check_flipped(angles)),
            #     "north: {:.2f})".format(
            #         north_deg),
            #     np.array(controller.drone_node.getPosition())
            #     )

        if show:
            cv2.destroyAllWindows()

    # run controller
    try:
        controller = WebotsSimulation()
        run(controller)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        controller.reset()
