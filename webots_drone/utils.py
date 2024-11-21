"""
Created on Sun Feb 26 21:49:14 2023

@author: Angel Ayala
"""
import cv2
import json
import base64
import numpy as np


def compute_orientation(point1, point2):
    # Calculate the differences in x and y coordinates
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    # Calculate the orientation (angle) using arctangent
    return np.arctan2(delta_y, delta_x)


def compute_risk_distance(fire_heigth, fire_radius):
    """Compute the risk zone distance between the drone and fire.
    The risk zone is consider 4 times the fire height as mentioned in
    Firefighter Safety Zones: A Theoretical Model Based on Radiative
    Heating, Butler, 1998.

    :param float fire_heigth: Indicate the fire's height.
    :param float fire_radius: Indicate the fire's radius.
    """
    return 4 * fire_heigth + fire_radius


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
        to 2 decimal points.
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2)).round(4)


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
    """Check if the uav_pos is outside the flight_area for ENU system."""
    # X axis check
    east = uav_pos[0] > flight_area[1][0]
    west = uav_pos[0] < flight_area[0][0]
    # Y axis check
    north = uav_pos[1] > flight_area[1][1]
    south = uav_pos[1] < flight_area[0][1]
    # Z axis check
    up = uav_pos[2] > flight_area[1][2]
    down = uav_pos[2] < flight_area[0][2]
    return [north, south, east, west, up, down]


def check_flipped(orientation, dist_sensors):
    """Check if the Drone was flipped.

    :param list orientation: The Drone's 3D orientation angles.
    :param list dist_sensors: The Drone's 3D normalized distance sensors.

    :return bool: True if have the propellers down side on the floor.
    """
    return (3. < orientation[0] or orientation[0] < -3.
            ) and dist_sensors[-1] < 0.01


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


def flight_area_norm_position(position, flight_area):
    position_norm = np.zeros_like(position, dtype=np.float32)
    for i, coord in enumerate(position):
        position_norm[i] = min_max_norm(coord, -1, 1,
                                        flight_area[0][i], flight_area[1][i])
    return position_norm


def target_mask(observation):
    """Compute the visual mask for red objects."""
    img_hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)
    img_hsv = cv2.GaussianBlur(img_hsv, (7, 7), 0)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # info
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    area = -1
    cpoint = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            M = cv2.moments(cnt)
            cpoint = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            break

    return mask, area, cpoint


def angle_90deg_offset(angle):
    """Apply UAV sensor offset."""
    angle -= np.pi / 2.
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle

def angle_inverse(angle):
    return -angle


def compute_target_orientation(position, ref_position):
    angle = compute_orientation(position, ref_position)
    angle = angle_inverse(angle_90deg_offset(angle))
    return angle


def check_target_distance(distance, distance_target, distance_margin=5.):
    area_limits = (distance_target - distance_margin / 2.,
                   distance_target + distance_margin / 2.)
    in_risk = distance < area_limits[0]
    in_zone = area_limits[0] <= distance < area_limits[1]
    out_zone = area_limits[1] <= distance
    return in_risk, in_zone, out_zone


def check_same_position(pisition1, position2, thr=0.003):
    dist_diff = compute_distance(pisition1, position2)
    dist_diff *= np.abs(dist_diff).round(4) > thr
    return dist_diff == 0.


def constrained_action(action, position, north_rad, flight_area, is_vel=False):
    """Check drone position and orientation to keep it inside flight_area."""
    # if velocity control swap roll and pitch values
    if is_vel:
        pitch_angle, roll_angle, yaw_angle, altitude = action
        roll_angle = -roll_angle
    else:
        roll_angle, pitch_angle, yaw_angle, altitude = action

    # check area limits
    out_area = check_flight_area(position, flight_area)

    # essential flags
    is_north = np.pi / 2. > north_rad > -np.pi / 2.  # north
    is_east = north_rad > 0
    orientation = [is_north,        # north
                   not is_north,    # south
                   is_east,         # east
                   not is_east]     # west
    movement = [pitch_angle > 0.,  # north - forward
                pitch_angle < 0.,  # south - backward
                roll_angle > 0.,   # east - right
                roll_angle < 0.]   # west - left

    # constraint action values
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

    if is_vel:
        return pitch_angle, -roll_angle, yaw_angle, altitude
    else:
        return roll_angle, pitch_angle, yaw_angle, altitude
