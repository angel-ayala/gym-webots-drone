#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:11:52 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import gym
import cv2
import numpy as np

from gym import spaces, logger
from gym.utils import seeding

from webots_drone import WebotsSimulation
from webots_drone.utils import info2state
from webots_drone.utils import min_max_norm
from webots_drone.utils import compute_distance
from webots_drone.utils import check_flight_area
from webots_drone.utils import check_collision
from webots_drone.utils import check_flipped
from webots_drone.utils import check_near_object


class DroneEnvContinuous(gym.Env):
    """Gym enviroment to control the Fire scenario in the Webots simulator."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 30
    }

    def __init__(self, time_limit=7500,  # 1 min
                 max_no_action_steps=625,  # 5 sec
                 frame_skip=125,  # 1 sec
                 goal_threshold=5.,
                 init_altitude=25.,
                 altitude_limits=[11, 75],
                 fire_pos=[-40, 40],
                 fire_dim=[7, 5],
                 is_pixels=True):
        # Simulation controller
        logger.info('Checking Webots connection...')
        self.sim = WebotsSimulation()
        logger.info('Connected to Webots')

        # Action space, the angles and altitud
        self.action_space = spaces.Box(low=self.sim.limits[0],
                                       high=self.sim.limits[1],
                                       shape=(self.sim.limits.shape[-1], ),
                                       dtype=np.float32)
        # Observation space
        self.is_pixels = is_pixels
        if self.is_pixels:
            # Observation space, the drone's camera image
            self.obs_shape = (3, 84, 84)
            self.obs_type = np.uint8
            self.observation_space = spaces.Box(low=0,
                                                high=255,
                                                shape=self.obs_shape,
                                                dtype=self.obs_type)
        else:
            self.obs_shape = (12, )
            self.obs_type = np.float32
            self.observation_space = spaces.Box(low=float('-inf'),
                                                high=float('inf'),
                                                shape=self.obs_shape,
                                                dtype=self.obs_type)
        # runtime vars
        self.init_runtime_vars()
        self._max_episode_steps = time_limit
        self._max_no_action_steps = round(max_no_action_steps / frame_skip)
        self._frame_skip = frame_skip
        self._reward_lim = [-200 - 50 * (frame_skip - 1), 100 * frame_skip]
        self._min_goal_distance = self.sim.risk_distance + goal_threshold
        self._fire_pos = fire_pos
        self._fire_dim = fire_dim
        self.flight_area = self.sim.get_flight_area(altitude_limits)
        self.init_altitude = init_altitude

        self.viewer = None

    def init_runtime_vars(self):
        self._episode_steps = 0  # time limit control
        self._no_action_steps = 0  # no action control
        self._end = False  # episode end flag
        self._prev_distance = 0  # reward helper
        self.last_info = dict()
        self.last_state = None

    def seed(self, seed=None):
        """Set seed for the environment random generations."""
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. Taken from atari_env.py
        # seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed2 = (seed1 + 1) % 2**31
        self.sim.seed(seed2)
        return [seed1, seed2]

    def preprocess_image(self, img):
        """BGR2RGB, center crop, and 84x84 resize operations."""
        # make it square from center
        hheight = self.sim.image_shape[0] // 2  # half height
        hcenter = self.sim.image_shape[1] // 2  # half width
        center_idx = (hcenter - hheight, hcenter + hheight)
        result = np.asarray(img[:, center_idx[0]:center_idx[1], :3],
                            dtype=self.obs_type)
        # resize
        result = cv2.resize(result, self.obs_shape[1:],
                            interpolation=cv2.INTER_AREA)
        # channel first
        result = np.swapaxes(result, 2, 0)
        return result

    def get_state(self):
        """Process the image to get a RGB image."""
        state_data = self.sim.get_data()
        image_rgb = state_data['image'].copy()[:, :, [2, 1, 0]]  # BGR2RGB
        if self.is_pixels:
            state = self.preprocess_image(image_rgb)
            del state_data['image']
        else:
            state = info2state(state_data)
            state_data['image_rgb'] = image_rgb

        return state, state_data

    def constrained_action(self, action):
        """Check drone position and orientation to keep inside FlightArea."""
        # clip action values
        action_clip = np.clip(action, self.sim.limits[0], self.sim.limits[1])
        roll_angle, pitch_angle, yaw_angle, altitude = action_clip

        info = self.sim.get_data()
        # check area contraints
        north_deg = info["north_deg"]
        out_area = check_flight_area(info["position"], self.flight_area)

        orientation = [north_deg > 0.5 or north_deg < -0.5,     # north
                       north_deg < 0.5 and north_deg > -0.5,    # south
                       north_deg > 0.,                          # east
                       north_deg < 0.]                          # west
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

    def __no_action_limit(self, position):
        # diff_pos = compute_distance(position, self.last_state[:3])
        if len(self.last_info.keys()) == 0:
            return False
        diff_pos = compute_distance(position, self.last_info['position'])
        if diff_pos <= 0.01:
            self._no_action_steps += 1 / self._frame_skip
        else:
            self._no_action_steps = 0
        return self._no_action_steps >= self._max_no_action_steps

    def __is_final_state(self, info, curr_distance):
        discount = 0
        end = False
        # no action limit
        if self.__no_action_limit(info["position"]):
            logger.info(f"[{info['timestamp']}] Final state, Same position")
            discount -= 200 * self._frame_skip
            end = True
            info['final'] = 'No Action'
        # is_flipped
        elif check_flipped(info["orientation"]):
            logger.info(f"[{info['timestamp']}] Final state, Flipped")
            discount -= 200
            end = True
            info['final'] = 'Flipped'
        # is_collision
        elif any(check_collision(info["dist_sensors"])):
            logger.info(f"[{info['timestamp']}] Final state, Collision")
            discount -= 200
            end = True
            info['final'] = 'Colission'
        # risk zone trespassing
        elif curr_distance < self.sim.risk_distance:
            logger.info(f"[{info['timestamp']}] Final state, Risk zone")
            discount -= 200
            end = True
            info['final'] = 'Risk zone'

        return discount, end

    def __compute_penalization(self, info):
        near_object_threshold = [150 / 4000,  # front left
                                 150 / 4000,  # front right
                                 150 / 3200,  # rear top
                                 150 / 3200,  # rear bottom
                                 150 / 1000,  # left side
                                 150 / 1000,  # right side
                                 150 / 2200,  # down front
                                 150 / 2200,  # down back
                                 30 / 800]  # top
        penalization = 0
        penalization_str = ''
        # object_near
        if any(check_near_object(info["dist_sensors"],
                                 near_object_threshold)):
            penalization -= 25
            penalization_str += 'object_near|'
        # outside flight area
        if any(check_flight_area(info["position"], self.flight_area)):
            penalization -= 50
            penalization_str += 'out_flight_area|'

        if len(penalization_str) > 0:
            info['penalization'] = penalization_str
        return penalization

    def compute_reward(self, obs, info):
        """Compute the distance-based reward.

        Compute the distance between drone and fire.
        This consider a risk_zone to 4 times the fire height as mentioned in
        Firefighter Safety Zones: A Theoretical Model Based on Radiative
        Heating, Butler, 1998.

        :param float distance_threshold: Indicate the acceptable distance
            margin before the fire's risk zone.
        """
        info['penalization'] = 'no'
        info['final'] = 'no'
        curr_distance = self.sim.get_target_distance()
        # terminal states
        discount, end = self.__is_final_state(info, curr_distance)
        if end:
            self._end = end
            return discount

        # not terminal, must be avoided
        penalization = self.__compute_penalization(info)
        if penalization > 0:
            return penalization

        reward = -1.0 + self._prev_distance - curr_distance
        self._prev_distance = curr_distance
        # goal achieved
        if curr_distance < self._min_goal_distance:
            reward = 100

        return reward

    def __time_limit(self):
        # time limit control
        self._episode_steps += 1
        return self._episode_steps >= self._max_episode_steps

    def perform_action(self, action):
        constrained_action = self.constrained_action(action)
        # perform action
        command = dict(disturbances=constrained_action)
        self.sim.send_data(command)
        # read state
        observation, info = self.get_state()
        # compute reward
        reward = self.compute_reward(observation, info)  # obtain reward
        # normalize reward
        reward = min_max_norm(reward, a=-1, b=1,
                              minx=self._reward_lim[0],
                              maxx=self._reward_lim[1])
        return observation, reward, info

    def lift_uav(self, altitude):
        diff_altitude = float('inf')
        lift_action = [0., 0., 0., self.sim.limits[1][3]]
        logger.info("Lifting the drone...")
        while diff_altitude > 0:
            _, _, info = self.perform_action(lift_action)
            diff_altitude = altitude - info['position'][2]  # Z axis diff
        logger.info("Drone lifted")

        self.perform_action([0., 0., 0., 0.])  # no action

    def reset(self, seed=None, options=None):
        """Reset episode in the Webots simulation."""
        # restart simulation
        self.seed(seed)
        self.sim.reset()
        # randomize fire position
        self.sim.randomize_fire_position()
        self.sim.play_fast()
        self.sim.sync()
        self.init_runtime_vars()
        self.lift_uav(self.init_altitude)
        self.last_state, self.last_info = self.get_state()

        return self.last_state, self.last_info

    def step(self, action):
        """Perform an action step in the simulation scene."""
        reward = 0
        for _ in range(self._frame_skip):
            observation, obs_reward, info = self.perform_action(action)
            reward += obs_reward  # step reward
            if self._end:
                break

        # timeout limit
        if self.__time_limit():
            logger.info(f"[{info['timestamp']}] Final state, Time limit")
            self._end = True
            info['final'] = 'time_limit'

        self.last_state, self.last_info = observation, info

        return self.last_state, reward, self._end, False, self.last_info

    def render(self, mode='human'):
        """Render the environment from Webots simulation."""
        if mode == 'rgb_array':
            return self.last_info['image']

        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()

            if 'image' in self.last_info.keys():
                self.viewer.imshow(self.last_info['image'])

            return self.viewer.isopen

    def close(self):
        """Close the environment and stop the simulation."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
