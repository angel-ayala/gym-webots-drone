"""
Created on Sun Feb 26 21:49:14 2023

@author: Angel Ayala
"""
import cv2
import json
import base64
import numpy as np


def preprocess_orientation(orientation):
    # Convert from [-pi, pi] to [0, 2pi]
    if orientation < 0:
        orientation += 2 * np.pi
    return orientation


def info2state(info):
    state = np.zeros((12, ), dtype=np.float32)
    if info is not None:
        state[:3] = info['position']  # position
        state[3:6] = info['orientation']  # orientation angles
        state[6:9] = info['speed']  # pos_vel
        state[9:] = info['angular_velocity']  # angular velocity
    return state


def emitter_send_json(emitter, data):
    str_data = json.dumps(data).encode('utf-8')
    emitter.send(str_data, len(str_data) + 1)


def receiver_get_json(receiver):
    command = dict()
    info = dict()
    if receiver.getQueueLength() > 0:
        msg = receiver.getString()[:-1]
        command = json.loads(msg)
        info = dict(
            direction=[receiver.getEmitterDirection()[0],
                       receiver.getEmitterDirection()[1],
                       receiver.getEmitterDirection()[2]],
            signal_strength=receiver.getSignalStrength())
        receiver.nextPacket()
    return command, info


def min_max_norm(x, a=0, b=1, minx=0, maxx=1):
    """Normalize a x integer in the [a, b] range.

    :param integer x: Number to be normalized.
    :param integer a: Low limit. The default is 0.
    :param integer b: upper limit. The default is 1.

    :return float, the number normalized.
    """
    return a + (((x - minx) * (b - a)) / (maxx - minx))


def compute_distance(coord1, coord2):
    """Compute squared Euclidean distance.

    :param np.array coord1: the first coordinates.
    :param np.array coord2: the second coordinates.

    :return np.array: squared difference sum of the coordinates, rounded
        to 4 decimal points.
    """
    c1 = np.asarray(coord1)
    c2 = np.asarray(coord2)
    return np.sqrt(np.sum(np.square(c1 - c2))).round(4)


def bytes2image(buffer, shape=(240, 400, 4)):
    """Translate a buffered image Camera() node into an BGRA channels array.

    :param bytes buffer: the buffer data of the image
    :param tuple size: (height, width, channels) of the image
    """
    array_image = np.frombuffer(buffer, np.uint8).reshape(shape)  # BGRA image
    return array_image


def encode_image(image):
    _, buffer_img = cv2.imencode('.png', image)
    encoded_data = base64.b64encode(buffer_img).decode('utf-8')
    return encoded_data


def decode_image(raw_image):
    buffer = base64.b64decode(raw_image)
    buffer_img = np.frombuffer(buffer, np.uint8)
    array_image = cv2.imdecode(buffer_img, cv2.IMREAD_UNCHANGED)
    return array_image

def check_flight_area(uav_pos, flight_area):
    """Check if the uav_pos is outside the flight_area."""
    # X axis check
    north = uav_pos[0] > flight_area[1][0]
    south = uav_pos[0] < flight_area[0][0]
    # Y axis check
    east = uav_pos[1] < flight_area[0][1]
    west = uav_pos[1] > flight_area[1][1]
    # Z axis check (altitude)
    down = uav_pos[2] < flight_area[0][2]
    up = uav_pos[2] > flight_area[1][2]
    return [north, south, east, west, up, down]

def check_flipped(angles):
    """Check if the Drone was flipped.

    :param list angles: The Drone's 3D normalized angles.

    :return bool: True if have the propellers down side.
    """
    # TODO: check value
    return angles[0] > 0.3  # top dist infrared


def check_near_object(sensors, threshold=0.01):
    """Check if the sensors encountered an object near.

    :param list sensors: The Drone's normalized sensors values.

    :return bool: True if the value of any sensor is below the threshold.
    """
    if type(threshold) is float:
        object_near = [sensor < threshold for sensor in sensors]
    if type(threshold) is list:
        object_near = [sensor < thr
                       for sensor, thr in zip(sensors, threshold)]
    else:
        raise ValueError("threshold value must be list or float value.")

    return object_near


def check_collision(sensors):
    collision_threshold = [50 / 4000,  # front left
                           50 / 4000,  # front right
                           50 / 3200,  # rear top
                           50 / 3200,  # rear bottom
                           50 / 1000,  # left side
                           50 / 1000,  # right side
                           50 / 2200,  # down front
                           50 / 2200,  # down back
                           10 / 800]  # top
    # is_collision
    near_object = check_near_object(sensors, collision_threshold)
    # filter sensors at the bottom
    return near_object[:6] + near_object[-1:]

def compute_flight_area(rc_pos, offset):
    flight_area = [
        [rc_pos[0] - offset[0][0],
         rc_pos[1] - offset[0][1],
         offset[0][2],],
        [rc_pos[0] + offset[1][0],
         rc_pos[1] + offset[1][1],
         offset[1][2],]
        ]
    return flight_area
