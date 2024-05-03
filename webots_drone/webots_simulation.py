#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:19:44 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import numpy as np
import traceback
import os
import sys

from webots_drone.utils import check_flight_area
from webots_drone.utils import compute_distance
from webots_drone.utils import min_max_norm
from webots_drone.utils import decode_image
from webots_drone.utils import emitter_send_json
from webots_drone.utils import receiver_get_json
from webots_drone.utils import compute_risk_distance

sys.path.append(os.environ['WEBOTS_HOME'] + "/lib/controller/python")
from controller import Supervisor


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
        self.limits = self.get_control_ranges()
        # runtime vars
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

    @staticmethod
    def get_control_ranges():
        """The control limits to manipulate the angles and altitude."""
        control_ranges = np.array([np.pi / 12.,     # roll
                                   np.pi / 12.,     # pitch
                                   np.pi,           # yaw
                                   5.               # altitude
                                   ])
        return np.array([control_ranges * -1,  # low limits
                         control_ranges])      # high limits

    def get_flight_area(self, altitude_limits=[11, 75]):
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
        # Forest area
        forest_shape = self.getFromDef('ForestArea').getField('shape')
        self.forest_area = []

        for i in range(forest_shape.getCount()):
            self.forest_area.append(forest_shape.getMFVec2f(i))
        self.forest_area = np.asarray(self.forest_area)

        return self.forest_area

    def init_target_node(self):
        # Fire vars
        target_node = self.getFromDef('Target')
        self.target_node = dict(
            node=target_node,
            get_height=lambda: target_node.getField('height').getSFFloat(),
            get_radius=lambda: target_node.getField('radius').getSFFloat(),
            get_pos=lambda: np.array(
                target_node.getField('translation').getSFVec3f()),
            set_height=target_node.getField('height').setSFFloat,
            set_radius=target_node.getField('radius').setSFFloat,
            set_pos=target_node.getField('translation').setSFVec3f
        )
        self.risk_distance = compute_risk_distance(
            self.target_node['get_height'](), self.target_node['get_radius']())

    def init_drone_node(self):
        # Drone vars
        drone_node = self.getFromDef('Drone')
        self.drone_node = dict(
            node=drone_node,
            get_pos=lambda: np.array(
                drone_node.getField('translation').getSFVec3f()),
            set_pos=drone_node.getField('translation').setSFVec3f
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
            #self.target_node['node'].restartController()
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

    def set_fire_dimension(self, fire_height=None, fire_radius=None):
        """
        Set the FireSmoke Node's height and radius.

        :param float fire_height: The fire's height, default is 2.
        :param float fire_radius: The fire's radius, default is 0.5

        :return float, float: the settled height and radius values.
        """
        if fire_height is None:
            fire_height = self.np_random.uniform(2., 13.)
        if fire_radius is None:
            fire_radius = self.np_random.uniform(0.5, 3.)

        # FireSmoke node fields
        self.target_node['set_height'](float(fire_height))
        self.target_node['set_radius'](float(fire_radius))

        # correct position in Z axis and update risk_distance value
        fire_pos = self.target_node['get_pos']()
        fire_pos[2] = fire_height * 0.5  # update height
        self.target_node['set_pos'](list(fire_pos))
        self.risk_distance = compute_risk_distance(fire_height, fire_radius)

        return (fire_height, fire_radius), self.risk_distance

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
        fire_p = self.target_node['get_pos']()  # current position
        # get forest limits
        X_range = [self.forest_area[:, 0].min(), self.forest_area[:, 0].max()]
        Y_range = [self.forest_area[:, 1].min(), self.forest_area[:, 1].max()]
        if fire_pos is None:
            # randomize position
            fire_p[0] = self.np_random.uniform(fire_radius - abs(X_range[0]),
                                               X_range[1] - fire_radius)
            fire_p[1] = self.np_random.uniform(fire_radius - abs(Y_range[0]),
                                               Y_range[1] - fire_radius)
        else:
            fire_p[0] = fire_pos[0]
            fire_p[1] = fire_pos[1]

        # ensure fire position inside the forest
        fire_p[0] = np.clip(fire_p[0], X_range[0], X_range[1])
        fire_p[1] = np.clip(fire_p[1], Y_range[0], Y_range[1])

        # set new position
        self.target_node['set_pos'](list(fire_p))

        return fire_p

    def set_fire(self, fire_pos=None, fire_height=None, fire_radius=None,
                 dist_threshold=0.):
        self.set_fire_dimension(fire_height, fire_radius)
        new_fire_pos = self.set_fire_position(fire_pos)

        # avoid to the fire appears near the drone's initial position
        must_do = 0
        directions = [(-1, 1), (1, 1),
                      (1, -1), (-1, -1)]
        while self.get_target_distance() <= self.get_risk_distance(dist_threshold):
            # randomize position offset
            offset = self.np_random.uniform(0.1, 1.)
            new_fire_pos[0] += offset * directions[must_do % 4][0]
            new_fire_pos[1] += offset * directions[must_do % 4][1]
            must_do += 1
            self.set_fire_position(new_fire_pos)

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

    def get_risk_distance(self, threshold=0.):
        return self.risk_distance + threshold

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
        north_rad = uav_state['north']
        dist_sensors = list()

        # Normalize distance sensor values
        for idx, sensor in uav_state['dist_sensors'].items():
            if sensor[2] == sensor[1] == sensor[0] == 0.:
                continue
            s_val = min_max_norm(sensor[0],
                                 a=0, b=1,
                                 minx=sensor[1], maxx=sensor[2])
            dist_sensors.append(s_val)

        if type(uav_state['image']) == str and uav_state['image'] == "NoImage":
            img = np.zeros(self.image_shape)
        else:
            img = decode_image(uav_state['image'])

        self._data = dict(timestamp=timestamp,
                          orientation=orientation,
                          angular_velocity=angular_velocity,
                          position=position,
                          speed=speed,
                          north_rad=north_rad,
                          dist_sensors=dist_sensors,
                          motors_vel=uav_state['motors_vel'],
                          image=img,
                          emitter=emitter_info,
                          rc_position=self.getSelf().getPosition(),
                          target_position=self.target_node['get_pos'](),
                          target_dim=[self.target_node['get_height'](),
                                      self.target_node['get_radius']()])

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

    def clip_action(self, action, flight_area):
        """Check drone position and orientation to keep inside FlightArea."""
        # clip action values
        action_clip = np.clip(action, self.limits[0], self.limits[1])
        roll_angle, pitch_angle, yaw_angle, altitude = action_clip

        # check area contraints
        info = self.get_data()
        north_rad = info["north_rad"]
        out_area = check_flight_area(info["position"], flight_area)

        is_north = np.pi / 2. > north_rad > -np.pi / 2.  # north
        is_east = north_rad < 0
        orientation = [is_north,        # north
                       not is_north,    # south
                       is_east,         # east
                       not is_east]     # west
        movement = [pitch_angle > 0.,  # north - forward
                    pitch_angle < 0.,  # south - backward
                    roll_angle > 0.,   # east - right
                    roll_angle < 0.]   # west - left

        if out_area[0]:
            if ((orientation[0] and movement[0])
                    or (orientation[1] and movement[1])):  # N,S
                pitch_angle = 0.

            if ((orientation[2] and movement[3])
                    or (orientation[3] and movement[2])):  # E,W
                roll_angle = 0.

        if out_area[1]:
            if ((orientation[0] and movement[1])
                    or (orientation[1] and movement[0])):  # N,S
                pitch_angle = 0.

            if ((orientation[2] and movement[2])
                    or (orientation[3] and movement[3])):  # E,W
                roll_angle = 0.

        if out_area[2]:
            if ((orientation[0] and movement[2])
                    or (orientation[1] and movement[3])):  # N,S
                roll_angle = 0.

            if ((orientation[2] and movement[0])
                    or (orientation[3] and movement[1])):  # E,W
                pitch_angle = 0.

        if out_area[3]:
            if ((orientation[0] and movement[3])
                    or (orientation[1] and movement[2])):  # N,S
                roll_angle = 0.

            if ((orientation[2] and movement[1])
                    or (orientation[3] and movement[0])):  # E,W
                pitch_angle = 0.

        if ((out_area[4] and altitude > 0)  # ascense
                or (out_area[5] and altitude < 0)):  # descense
            altitude = 0.

        return roll_angle, pitch_angle, yaw_angle, altitude


if __name__ == '__main__':
    import cv2
    import datetime
    from webots_drone.reward import compute_vector_reward
    # from webots_drone.utils import compute_target_orientation
    # from webots_drone.envs.preprocessor import info2image
    # from webots_drone.reward import compute_visual_reward

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

    def get_kb_action(kb):
        # capture control data
        key = kb.getKey()

        run_flag = True
        take_shot = False
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
                pitch_angle = controller.limits[1][1]
            elif key == kb.DOWN:
                pitch_angle = controller.limits[0][1]
            # yaw
            elif key == ord('D'):
                yaw_angle = controller.limits[0][2]
            elif key == ord('A'):
                yaw_angle = controller.limits[1][2]
            # altitude
            elif key == ord('W'):
                altitude = controller.limits[1][3]  # * 0.1
            elif key == ord('S'):
                altitude = controller.limits[0][3]  # * 0.1
            # quit
            elif key == ord('Q'):
                print('Terminated')
                run_flag = False
            # take photo
            elif key == ord('P'):
                print('Camera frame saved')
                take_shot = True
            key = kb.getKey()

        action = [roll_angle, pitch_angle, yaw_angle, altitude]
        action = controller.clip_action(action, controller.get_flight_area())
        return action, run_flag, take_shot

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
        goal_threshold = 5.
        fire_pos = [50, -50]
        fire_dim = [7., 3.5]
        altitude_limits = [11., 75.]
        controller.seed()
        controller.set_fire(fire_pos=fire_pos, fire_height=fire_dim[0],
                            fire_radius=fire_dim[1],
                            dist_threshold=goal_threshold)
        controller.play()
        controller.sync()
        run_flag = True
        take_shot = False
        frame_skip = 25
        step = 0
        accum_reward = 0
        # capture initial state
        state = controller.get_data()
        next_state = controller.get_data()

        print('Fire scene is running')
        while (run_flag):  # and drone.getTime() < 30):
            # 2 dimension considered
            uav_pos_t = state['position'][:2]  # pos_t
            uav_pos_t1 = next_state['position'][:2]  # pos_t+1
            uav_ori_t1 = next_state['north_rad']
            target_xy = controller.get_target_pos()[:2]
            # target_ori = compute_target_orientation(uav_pos_t1, target_xy)

            # compute reward components
            curr_distance = compute_distance(target_xy, uav_pos_t)
            next_distance = compute_distance(target_xy, uav_pos_t1)
            distance_diff = np.round(curr_distance - next_distance, 3)
            distance_diff *= np.abs(distance_diff) > 0.003
            reward = compute_vector_reward(
                target_xy, uav_pos_t, uav_pos_t1, uav_ori_t1,
                distance_target=controller.get_risk_distance(goal_threshold / 2.),
                distance_margin=goal_threshold)

            # observation = info2image(next_state, output_size=84)
            # reward += compute_visual_reward(observation)
            accum_reward += reward
            if step % frame_skip == 0:
                print(f"pos_t: {uav_pos_t[0]:.3f} {uav_pos_t[1]:.3f}"\
                      f" - post_t+1: {uav_pos_t1[0]:.3f} {uav_pos_t1[1]:.3f} (N:{uav_ori_t1:.3f})"\
                      f" -> reward ({reward:.4f}) {accum_reward:.4f}"\
                      f" diff: {distance_diff:.4f} ({curr_distance:.4f}/{controller.get_risk_distance(goal_threshold/2.):.4f})")
                accum_reward = 0

            if take_shot:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                cv2.imwrite(f'photos/picture_{timestamp}.png', state['image'])

            state = next_state.copy()
            # capture action
            action, run_flag, take_shot = get_kb_action(kb)
            disturbances = dict(disturbances=action)
            # perform action
            controller.send_data(disturbances)
            # capture state
            next_state = controller.get_data()
            if show and next_state is not None:
                cv2.imshow("Drone's live view", next_state['image'])
                cv2.waitKey(1)
            step += 1
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
