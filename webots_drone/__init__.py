"""
webots_drone.

A gym wrapper for Webots simulations scene with the DJI Mavic Pro 2 drone.
"""
from gym.envs.registration import register
from .webots_simulation import WebotsSimulation


__all__ = ['WebotsSimulation']
__version__ = "0.1.0"
__author__ = 'Angel Ayala'


register(
    id='webots_drone/DroneEnvContinuous-v0',
    entry_point='webots_drone.envs:DroneEnvContinuous',
)

register(
    id='webots_drone/DroneEnvDiscrete-v0',
    entry_point='webots_drone.envs:DroneEnvDiscrete',
)
