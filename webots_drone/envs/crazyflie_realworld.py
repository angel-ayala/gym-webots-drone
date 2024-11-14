#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 04 09:56:27 2024

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
from typing import Union
import numpy as np
from gym import logger
from pathlib import Path

from webots_drone.envs import CrazyflieEnvContinuous


class RealCrazyflieEnvContinuous(CrazyflieEnvContinuous):
    """Gym enviroment to control the Crazyflie drone in Webots simulator."""

    def __init__(self, agent_id,
                 timestep=32,  # ms
                 time_limit_seconds=60,  # 1 min
                 max_no_action_seconds=5,  # 5 sec
                 frame_skip=6,  # 1 sec
                 goal_threshold=0.5,
                 init_altitude=1.,
                 altitude_limits=[0.25, 2.],
                 fire_pos=2,
                 fire_dim=[.05, .02],
                 is_pixels=False,
                 zone_steps=10):

        self.init_altitude = init_altitude
        self.agent_id = agent_id
        self.timestep = timestep

        super(RealCrazyflieEnvContinuous, self).__init__(
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
        import sys
        # Change this path to the CrazyflieEnvironment folder
        sys.path.append('../../cf_client_env')
        from cf_virtual_env import CrazyflieEnvironment
        # Simulation controller
        logger.info('Checking Crazyflie connection...')
        visible_area = [
            [-1.3, -1.3, 0.25],  # lower limits
            [ 1., 1., 2.]   # higher limits
        ]
        self.sim = CrazyflieEnvironment(agent_id=self.agent_id,
                                        init_height=self.init_altitude,
                                        timestep=self.timestep,
                                        flight_area=visible_area)
        logger.info(f"Connected to {self.sim.drone}")

    def create_target(self, position=None, dimension=None):
        # virtualTarget
        from webots_drone.target import VirtualTarget
        if type(position) is int:
            position = self.quadrants[position]
        return VirtualTarget(position=position, dimension=dimension, is_3d=True)

    def get_state(self):
        """Process the environment to get a state."""
        state_data = self.sim.get_data()
        state_data['target_position'] = self.vtarget.position
        state_data['target_dim'] = self.vtarget.dimension
        # angular values from degree to radians
        state_data['orientation'] = np.radians(state_data['orientation'])
        state_data['angular_velocity'] = np.radians(state_data['angular_velocity'])
        state_data['north_rad'] = np.radians(state_data['north_deg'])
        del state_data['north_deg']
        # change the line below if any distance sensor is available
        state_data['dist_sensors'] = []

        state = self.get_observation_1d(state_data)

        return state, state_data

    def reset(self, seed=None, fire_quadrant=None, **kwargs):
        """Reset episode in the Webots simulation."""
        # restart simulation
        self.seed(seed)
        self.sim.reset()
        self.last_state, self.last_info = self.get_state()
        fquadrant = self._fire_pos if fire_quadrant is None else fire_quadrant
        self.set_fire_quadrant(fquadrant)
        self.init_runtime_vars()
        self.last_state, self.last_info = self.get_state()

        return self.last_state, self.last_info

    def step(self, action):
        """Perform an action step in the simulation scene."""
        observation, reward, end, truncated, info = super().step(action)

        if self.sim.drone.is_low_battery:
            logger.info(f"[{info['timestamp']}] Final state, Low Battery")
            truncated = True
            info['final'] = 'low_battery'

        return observation, reward, end, truncated, info

    def close(self):
        super().close()
        self.sim.close()


def print_state(state):
    # position
    x, y, z = state['position']
    state_str = f"pos({x:.3f}, {y:.3f}, {z:.3f})\t"
    # attitude
    roll, pitch, yaw = state['orientation']
    state_str += f"att({roll:.1f}, {pitch:.1f}, {yaw:.1f})\t"
    # battery
    state_str += f"bat({state['battery_volts']:.2f})"
    # TODO: add translational velocity
    # TODO: add angular velocity
    print('[StateInfo]:', state_str)


def run_episode(drone_env: RealCrazyflieEnvContinuous, ep: int = 0, logs_path: Union[Path, None] = None):
    observation, info = drone_env.reset()
    terminate = False
    while not terminate:
        # ------------------------- Capture control signal ---------------------------
        # vel_x, vel_y, rate_yaw, vel_z
        action = keyboard2action(drone_env.action_limits[1])

        # --------------- Send control signal to env ---------------------------
        observation, reward, end, truncated, info = drone_env.step(action)
        terminate = end or truncated
        # print_state(info)
        gpos = ", ".join([f"{v:.3f}" for v in info['target_position']])
        tdist = drone_env.vtarget.get_distance(info['position'])
        gdist = drone_env.distance_target
        print(f"R:{reward:.4f}\tGDist: {tdist:.3f} / {gdist:.3f} ({gpos})")

        if keyboard.is_pressed('space'):
            terminate = True

        if logs_path is not None:
            log_data = {'episode': ep,
                        'home_position': drone_env.sim.drone.home_pos}
            log_data.update(info.copy())
            log_data['action'] = action
            log_data['reward'] = reward
            log_data['end'] = end
            log_data['truncated'] = truncated
            log_data['dist_current'] = tdist
            log_data['dist_goal'] = gdist
            log_dict(logs_path, log_data)


if __name__ == '__main__':
    import keyboard
    import traceback
    import datetime
    from pathlib import Path
    from utils import keyboard2action
    from utils import log_dict

    logger.set_level(logger.DEBUG)

    episodes = 1

    try:
        # Summary log_file
        logs_dir = Path("./logs_env")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logs_path = logs_dir / f"GymCrazyflieEnv_{timestamp}.csv"

        # Instantiate environment
        drone_env = RealCrazyflieEnvContinuous(
            agent_id=8,
            timestep=32,  # ms
            time_limit_seconds=60,  # 1 min
            max_no_action_seconds=5,  # 5 sec
            frame_skip=6,  # 1 sec
            goal_threshold=0.5,
            # init_altitude=0.3,
            init_altitude=1.,
            altitude_limits=[0.25, 2.],
            fire_pos=2,
            fire_dim=[.05, .02],
            is_pixels=False,
            zone_steps=10)

        for ep in range(episodes):
            run_episode(drone_env, ep=ep, logs_path=logs_path)

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)

    finally:
        drone_env.close()