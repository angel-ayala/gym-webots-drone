# README

# Webots Drone Scene

This repository is a [Gym](https://github.com/openai/gym) environment for [Webots](https://github.com/cyberbotics/webots) drone scene focused on UAV navigation research. The current research project is focused in fire emergency outdoor simulated scenarios, based on previous [Gym wrapper](https://github.com/angel-ayala/gym-webots-fire) for a [forest fire scene](https://github.com/angel-ayala/webots-fire-scene). The environment is intended to train a Reinforcement
Learning agent to control a drone under a fire emergency context. In order to make the drone flight, the algorithm must be capable to work under a continuous domain for both action and state space. The agent’s state can be represented by an image, a vector state, or both from which must decide the action to take. The mission of the agent is to approach to the fire location keeping a safe distance.

## Updates

- 2024/02/09 [Major fixes] v1.2: update reward values $`\in [-1, 0]`$ and adding distance difference as a boolean factor
- 2024/01/30 [Major fixes] v1.1: multi modal approaches support, store data fixes

## Available environments

There are currently two environments which differs in the action’s domain space.

- `webots_drone/DroneEnvContinuous-v0` presents a continuous action space domain composed by a 4-elements vector represented by {$`\phi, \theta, \psi,`$ throttle} corresponding to roll, pitch, and yaw angles, and the altitude desired for the drone.
    - $`\phi`$ is related to the translation in x-axis moving the drone to the left or the right.
    - $`\theta`$ is related to the translation in y-axis moving the drone forward or backward.
    - $`\psi`$ is related to the rotation in z-axis and rotates the drone in counter- or clockwise directions.
    - throttle is related to the translation in z-axis and move the drone up or
    down.
- `webots_drone/DroneEnvDiscrete-v0` is an extension of `DroneEnvContinuous-v0` with a discrete action space domain composed by a 6 actions + 1 no-action posibilities. The action is
discretized and the same step logic from the continuous domain is applied. It is considered two actions for each continuous degree-of-freedom and a fixed altituted resulting in:
    - Action 0: no-action
    - Action 1: affects $`\phi`$ with $`\pi / 12 => [\pi / 12, 0, 0, 0]`$
    - Action 2: affects $`\phi`$ with $`-\pi / 12 => [-\pi / 12, 0, 0, 0]`$
    - Action 3: affects $`\theta`$ with $`\pi / 12 => [0, \pi / 12, 0, 0]`$
    - Action 4: affects $`\theta`$ with $`-\pi / 12 => [0, -\pi / 12, 0, 0]`$
    - Action 5: affects $`\psi`$ with $`\pi => [0, 0, \pi, 0]`$
    - Action 6: affects $`\psi`$ with $`-\pi => [0, 0, -\pi, 0]`$

The observation space is a high-dimensional image, represented by the drone’s $`400 \times 240`$ pixels BGRA channels camera image. The observation is processed to get an image with RGB channels and values $`\in [0, 255]`$.
Additionally, the observation space also includes a low-dimensional sensor readings such as IMU, Gyroscope, GPS, and Magnetometer.

## Reward function

~~The reward function is the Euclidean distance between the drone’s position and the safe zone edge, calculated in from the fire location (target position). The safe zone edge is defined at the fire location as base, add the radius size is 4 times the fire’s height. This reward function start with a under zero value, and increase while the drone is getting close of the fire location. If this value is great than zero, the episode’s end. The reward function is defined as follows:~~

The reward signal comprises three components regard the velocity, pose and bonus/penalty values. The following methods implements it:

```python
def compute_vector_reward(ref_position, pos_t, pos_t1, orientation_t1,
                          distance_target=36., distance_margin=5.):
    # compute orientation reward
    ref_orientation = compute_target_orientation(pos_t1, ref_position)
    r_orientation = orientation2reward(orientation_t1, ref_orientation)
    # compute distance reward
    dist_t1 = compute_distance(pos_t1, ref_position)
    r_distance = distance2reward(dist_t1, distance_target)
    # compute velocity reward
    dist_t = compute_distance(pos_t, ref_position)
    r_velocity = velocity2reward(dist_t - dist_t1)
    # check zones
    zones = check_target_distance(dist_t1, distance_target, distance_margin)
    # inverse when trespass risk distance
    if zones[0]:
        r_velocity *= -1.
    # bonus in-distance
    r_bonus = 0.
    if zones[1]:
        r_bonus = 3.
        r_velocity = compute_distance(pos_t1, pos_t) / 0.035
    # penalty no movement
    elif check_same_position(pos_t, pos_t1):
        r_bonus -= 2.
    # if r_velocity < 0.:
    #     r_velocity = r_velocity / 2.

    # compose reward
    r_velocity = r_velocity * r_orientation  # [-1, 1]
    r_pose = r_distance + r_orientation - 1  # ]-inf, 0]
    r_sum = r_velocity + r_pose * 0.1 + r_bonus
    return r_sum
```

Additionally some penalties were considered to ensure safety and energy efficiency. A ring zone delimitates the risk area which the UAV must avoid because can suffer damage. Very near to it there is the goal region which is the closest area to reach around the fire. A square area delimitates the allowed flight area to avoid the drone go far away. As safety must be asure to flight far enough of obstacles, adding a penalization if is near of any object or if collided with it. The following method implements it.

```python
def __compute_penalization(self, info, zones):
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
        logger.info(f"[{info['timestamp']}] Penalty state, ObjectNear")
        penalization -= 1.
        penalization_str += 'ObjectNear|'
    # is_collision
    if any(check_collision(info["dist_sensors"])):
        logger.info(f"[{info['timestamp']}] Penalty state, Near2Collision")
        penalization -= 2.
        penalization_str += 'Near2Collision|'
    # outside flight area
    if any(check_flight_area(info["position"], self.flight_area)):
        logger.info(f"[{info['timestamp']}] Penalty state, OutFlightArea")
        penalization -= 2.
        penalization_str = 'OutFlightArea|'
    # risk zone trespassing
    if zones[0]:
        logger.info(f"[{info['timestamp']}] Penalty state, InsideRiskZone")
        penalization -= 2.
        penalization_str += 'InsideRiskZone|'

    if len(penalization_str) > 0:
        info['penalization'] = penalization_str

    return penalization
```

If any penalization occurs, the penalty score is inmediately returned, on the contrary the computed distance and orientation signal is returned instead. Each value is normalized $`\in [0, 1]`$the are multiplied and normalized again. The following method implements it:

```python
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

    # 2 dimension considered
    if len(self.last_info.keys()) == 0:
        uav_pos_t = info['position'][:2]  # pos_t
    else:
        uav_pos_t = self.last_info['position'][:2]  # pos_t
    uav_pos_t1 = info['position'][:2]  # pos_t+1
    uav_ori_t1 = info['north_rad']  # orientation_t+1
    target_xy = self.sim.get_target_pos()[:2]

    # compute reward components
    reward = compute_vector_reward(
        target_xy, uav_pos_t, uav_pos_t1, uav_ori_t1,
        distance_target=self.distance_target,
        distance_margin=self._goal_threshold)

    # if self.is_pixels:
    #     reward += compute_visual_reward(obs)

    zones = check_target_distance(self.sim.get_target_distance(),
                                  self.distance_target,
                                  self._goal_threshold)
    # allow no action inside zone
    if zones[1]:
        self._in_zone_steps += 1
    else:
        self._in_zone_steps = 0

    # not terminal, must be avoided
    penalization = self.__compute_penalization(info, zones)
    if penalization < 0:
        reward += penalization

    # terminal states
    discount = self.__is_final_state(info, zones)
    if discount < 0:
        self._end = True
        reward += discount

    return reward
```

## Installation and use

In order to use this environment you require to download and install the [Webots simulation](https://cyberbotics.com/doc/guide/installation-procedure) first. Then you must clone the repository and install the repository via pip.

```
git clone https://github.com/angel-ayala/gym-webots-drone.git
cd gym-webots-drone
pip install -e .
```

Then open from Webots open the world scene located in `worlds/forest_tower.wbt`. If no error are shown in the Webots console, then you can test the scene connection. Inside your local copy of this repository do

```
python webots_drone/webots_simulation.py
```

You should be able to control the drone in the Webots scene with `w,a,s,d` and arrows keys, more details should appeared in the Webots console.

### Reinforcement lerning random agent example

```python
import gym
import time
env = gym.make('webots_drone:webots_drone/DroneEnvDiscrete-v0',
               time_limit_seconds=60,  # 1 min
               max_no_action_seconds=5,  # 30 sec
               frame_skip=25,  # 200 ms # 125,  # 1 sec
               goal_threshold=25.,
               init_altitude=25.,
               altitude_limits=[11, 75],
               is_pixels=False)
print('reset')
env.reset()
for _ in range(1250):
    env.render()
    action = env.action_space.sample()
    print('action', action)
    env.step(action) # take a random actionenv.close()
```

### Considerations of the environment

This environment is an interface for the Webots simulation software version [R2023b](https://github.com/cyberbotics/webots/tree/R2023b), and you should be able to run the gym environment after downloading and installing the simulation software version. One main aspects to highlight is to ensure that webots environment variables are set such as `WEBOTS_HOME` and `LD_LIBRARY_PATH`. In linux can be setted with:

```
export WEBOTS_HOME=/path/to/webots
```

In order to check:

```
echo $WEBOTS_HOME # must show /path/to/webots
```

Finally you can execute your code that implement this environment as
usual.

# Webots Drone Scene

A simulated Webots scene with a first approach of fire simulation through the `FireSmoke` PROTO file. In this scene the DJI Mavic 2 Pro drone is available to control, variating the roll, pitch and yaw angles, and the altitude. The scene is intended to run automously from the Webots interface using a Robot node with the supervisor option set to `TRUE` and with the controller set to `‘<extern>`’. The `WebotsSimulation` class is configured to run from a terminal, being capable to:

1. Randomize the size and location of the FireSmoke node.
2. Start the simulation.
3. Acquire, and optionally show the image from the drone’s camera.
4. Control the drone from keyboard inputs.
5. Stop and restart the simulation.

## The FireSmoke PROTO

This `PROTO` node is implemented as a Robot node, using two `Display` nodes for the fire and smoke image. The `FireMovement` controller manage this nodes using two sprite cheet images to simulate the movement of the fire and the smoke in a low resolution. A safe distance from the fire was implemented using a first approach as presented by Butler[1] which define the safe distance as 4 times the fire’s height. To achieve this, the fire location is used as base with a radius heat size 4 times the fire’s height. If the drone exceed this point is considered a final state and the episode will be restarted. This node still need consider more realistic aspect such as heat propragation and wind resistance. 

**TODO** 

- [x]  Fire and Smoke movement.
- [x]  Safe distance of heat.
- [ ]  Smoke cloud.
- [ ]  Heat propagation.

## The Drone’s controller

The Mavic 2 Pro control is achieved by the `DroneController` class which have a two-way communication with `WebotsSimulation` using an `Emitter` and `Receiver` Webots node at each side. This setup is intended to simulated a considered as a ground station radio control from the
tower, from where the drone will receive angles and thrust disturbances along communicates its current sensor values.

### Motors control

All the sensors and actuators of the drone are managed by the `Drone` class, which instantiate and controls all the drone’s sensors and actuators. The velocity of each propeller motor is calculated using four `PID` controlles, one for the roll, pitch and yaw angles, and another for the altitude. The controllers were tunned using the Ziegler-Nichols PID tunning technique.

**IMPORTANT!** the controllers PID values were tunned for a 8ms simulation timestep, with a configured `defaultDamping` `WorldInfo`’s field using a Damping node with 0.5 value for the angular and linear field.

### Camera image

The drone’s is equipped with a Camera Node (can be modified in the Webots scene) with an $`400 \times 240`$ pixels BGRA channel image over a 3-axis gimbal to smooth the image movement.

### References

- [1] Firefighter Safety Zones: A Theoretical Model Based on Radiative
Heating, Butler, 1998.
