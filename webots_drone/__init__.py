"""
webots_drone.

A gym wrapper for Webots simulations scene with the DJI Mavic Pro 2 and
Crazyflie drones.
"""

import os
from gym.envs.registration import register

# Simulated environment requires of Webots' environment variables
if "WEBOTS_HOME" in os.environ:
    from .webots_simulation import WebotsSimulation
    from .cf_simulation import CFSimulation

    __all__ = ['WebotsSimulation', 'CFSimulation']


__version__ = "2.1.0"
__author__ = 'Angel Ayala'


register(
    id='webots_drone/DroneEnvContinuous-v0',
    entry_point='webots_drone.envs:DroneEnvContinuous',
)

register(
    id='webots_drone/DroneEnvDiscrete-v0',
    entry_point='webots_drone.envs:DroneEnvDiscrete',
)

register(
    id='webots_drone/CrazyflieEnvContinuous-v0',
    entry_point='webots_drone.envs:CrazyflieEnvContinuous',
)

register(
    id='webots_drone/CrazyflieEnvDiscrete-v0',
    entry_point='webots_drone.envs:CrazyflieEnvDiscrete',
)

register(
    id='webots_drone/RealCrazyflieEnvContinuous-v0',
    entry_point='webots_drone.envs:RealCrazyflieEnvContinuous',
)
