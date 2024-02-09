# README

# Webots Drone Scene

This repository is a [Gym](https://github.com/openai/gym) environment for [Webots](https://github.com/cyberbotics/webots) drone scene focused on UAV navigation research. The current research project is focused in fire emergency outdoor simulated scenarios, based on previous [Gym wrapper](https://github.com/angel-ayala/gym-webots-fire) for a [forest fire scene](https://github.com/angel-ayala/webots-fire-scene). The environment is intended to train a Reinforcement
Learning agent to control a drone under a fire emergency context. In order to make the drone flight, the algorithm must be capable to work under a continuous domain for both action and state space. The agent’s state can be represented by an image, a vector state, or both from which must decide the action to take. The mission of the agent is to approach to the fire location keeping a safe distance.

## Updates

- 2024/02/09 [Major fixes] v1.2: update reward values $`\in [-1, 0]`$ and adding distance difference as a boolean factor
- 2024/01/30 [Major fixes] v1.1: multi modal approaches support, store data fixes

## Available environments

There are currently two environments which differs in the action’s domain space.

- `webots_drone/DroneEnvContinuous-v0` presents a continuous action space domain composed by a 4-elements vector represented by $`\{\phi, \theta, \psi, thrust\}`$ corresponding to roll, pitch, and yaw angles, and the altitude desired for the drone.
    - $`\phi`$ is related to the translation in x-axis moving the drone to the left or the right.
    - $`\theta`$ is related to the translation in y-axis moving the drone forward or backward.
    - $`\psi`$ is related to the rotation in z-axis and rotates the drone in counter- or clockwise directions.
    - $`thrust`$ is related to the translation in z-axis and move the drone in up or
    down.
- `webots_drone/DroneEnvDiscrete-v0` is an extension of `DroneEnvContinuous-v0` with a discrete action space domain composed by a 6 actions + 1 no-action posibilities. The action is
discretized and the same step logic from the continuous domain is applied. It is considered two actions for each continuous degree-of-freedom and a fixed altituted resulting in:
    - Action 0: no-action
    - Action 1: affects $`\phi`$ with $`\pi / 12 => [\pi / 12, 0, 0, 0]`$
    - Action 2: affects $`\theta`$ with $`\pi / 12 => [0, \pi / 12, 0, 0]`$
    - Action 3: affects $`\psi`$ with $`\pi => [0, 0, \pi, 0]`$
    - Action 4: affects $`\phi`$ with $`-\pi / 12 => [-\pi / 12, 0, 0, 0]`$
    - Action 5: affects $`\theta`$ with $`-\pi / 12 => [0, -\pi / 12, 0, 0]`$
    - Action 6: affects $`\psi`$ with $`-\pi => [0, 0, -\pi, 0]`$

The observation space is a high-dimensional image, represented by the drone’s $`400 \times 240`$ pixels BGRA channels camera image. The observation is processed to get an image with RGB channels and values $`\in [0, 255]`$.

## Reward function

~~The reward function is the Euclidean distance between the drone’s position and the safe zone edge, calculated in from the fire location (target position). The safe zone edge is defined at the fire location as base, add the radius size is 4 times the fire’s height. This reward function start with a under zero value, and increase while the drone is getting close of the fire location. If this value is great than zero, the episode’s end. The reward function is defined as follows:~~

The reward signal comprises a two component value of distance and orientation. The value distance is the difference between the target position and the UAV position with two threshold values corresponding to the range are of detection and the nearest it can reach. The following methods implements it:

```python
def compute_direction_vector(position, orientation, ref_position):
    # Calculate the vector pointing from the agent position to the target position
    vector_to_target = np.array(ref_position) - np.array(position)
    # Normalize the vector
    norm_vector_to_target = vector_to_target / np.linalg.norm(vector_to_target)
    # Calculate the agent's forward vector based on its orientation
    agent_forward_vector = np.array([np.cos(orientation),
                                     np.sin(orientation)])

    return norm_vector_to_target, agent_forward_vector

def compute_orientation_reward(position, orientation, ref_position):
    # Get direction vectors
    direction_to_target, agent_forward = compute_direction_vector(
        position, orientation, ref_position)

    # Calculate cosine similarity between the direction to the target and agent's forward direction
    cosine_similarity = np.dot(direction_to_target, agent_forward)

    return (cosine_similarity - 1.) / 2.

def compute_distance_reward(position, ref_position, distance_max=50.,
                            distance_threshold=25., threshold_offset=5.):
    curr_distance = compute_distance(position, ref_position)
    safety_distance = distance_threshold - threshold_offset / 2
    reward = 1 - abs(1 - curr_distance / safety_distance)
    reward = max(-1., reward)

    if curr_distance < distance_threshold - threshold_offset:
        return -1.

    return (reward - 1.) / 2.

def sum_and_normalize(orientation_rewards, distance_rewards, distance_diff=1.):
    r_distance = (distance_rewards + 1.)
    r_orientation = (orientation_rewards + 1.)
    r_sum = r_distance * r_orientation * (distance_diff != 0.)
    return r_sum - 1.
```

Additionally some penalties were considered to ensure safety and energy efficiency. A ring zone delimitates the risk area which the UAV must avoid because can suffer damage. Very near to it there is the goal region which is the closest area to reach around the fire. A square area delimitates the allowed flight area to avoid the drone go far away. As safety must be asure to flight far enough of obstacles, adding a penalization if is near of any object or if collided with it. The following method implements it.

```python
def __compute_penalization(self, info, curr_distance):
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
        logger.info(f"[{info['timestamp']}] Warning state, ObjectNear")
        penalization -= 10
        penalization_str += 'ObjectNear|'
    # outside flight area
    if any(check_flight_area(info["position"], self.flight_area)):
        logger.info(f"[{info['timestamp']}] Warning state, OutFlightArea")
        penalization -= 10
        penalization_str += 'OutFlightArea|'
    # is_collision
    if any(check_collision(info["dist_sensors"])):
        logger.info(f"[{info['timestamp']}] Warning state, Near2Collision")
        penalization -= 10
        penalization_str += 'Near2Collision|'
    # risk zone trespassing
    if curr_distance < self.sim.risk_distance:
        logger.info(f"[{info['timestamp']}] Warning state, InsideRiskZone")
        penalization -= 10
        penalization_str += 'InsideRiskZone'

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
    # terminal states
    discount, end = self.__is_final_state(info)
    if end:
        self._end = end
        return discount

    # 2 dimension considered
    if len(self.last_info.keys()) == 0:
        uav_pos_t = info['position'][:2]  # pos_t+1
    else:
        uav_pos_t = self.last_info['position'][:2]  # pos_t
    uav_pos_t1 = info['position'][:2]  # pos_t+1
    uav_ori = info['north_rad']
    target_xy = self.sim.get_target_pos()[:2]

    # not terminal, must be avoided
    goal_distance = compute_distance(uav_pos_t1, target_xy)
    penalization = self.__compute_penalization(info, goal_distance)
    if penalization != 0:
        return penalization

    # compute reward components
    orientation_reward = compute_orientation_reward(uav_pos_t, uav_ori,
                                                    target_xy)
    distance_reward = compute_distance_reward(
        uav_pos_t, target_xy, distance_max=50.,
        distance_threshold=np.sum(self._goal_threshold),
        threshold_offset=self._goal_threshold[1])

    distance_diff = compute_distance(uav_pos_t, uav_pos_t1)
    reward = sum_rewards(orientation_reward, distance_reward,
                         distance_diff=distance_diff)

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
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/controller
```

In order to check:

```
echo $WEBOTS_HOME # must show /path/to/webots
echo $LD_LIBRARY_PATH # must show /path/to/webots/lib/controller
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
