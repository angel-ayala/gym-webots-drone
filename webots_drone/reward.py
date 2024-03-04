#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:35:08 2023

@author: Angel Ayala
"""

import numpy as np
from webots_drone.utils import compute_distance
from webots_drone.utils import compute_orientation
from webots_drone.utils import min_max_norm


def compute_direction_vector(ref_position, position, orientation):
    # Calculate the vector pointing from the agent's position to the target position
    vector_to_target = np.array(ref_position) - np.array(position)

    # Normalize the vector
    norm_vector_to_target = vector_to_target / np.linalg.norm(vector_to_target)

    # Calculate the agent's forward vector based on its orientation
    agent_forward_vector = np.array([np.cos(orientation),
                                     np.sin(orientation)])

    return norm_vector_to_target, agent_forward_vector


def compute_orientation_reward(ref_position, position, orientation):
    # Get direction vectors
    direction_to_target, agent_forward = compute_direction_vector(
        ref_position, position, orientation)

    # Calculate cosine similarity between the direction to the target and agent's forward direction
    cosine_similarity = np.dot(direction_to_target, agent_forward)

    return cosine_similarity


def compute_target_orientation_reward(ref_position, position, orientation,
                                      threshold_offset=0.1, n_segments=3):
    curr_distance = compute_orientation_reward(ref_position, position,
                                               orientation)
    # r_distance = (curr_distance + 1.) / 2.
    r_distance = curr_distance
    # create a distance scale intervals
    for n in range(n_segments + 1):
        if curr_distance >= 1 - threshold_offset * n:
            r_distance *= 2.

    # r_distance = max(-1., r_distance)
    return r_distance


def sum_and_normalize(distance_rewards, orientation_rewards):
    r_distance = distance_rewards
    r_orientation = orientation_rewards
    r_sum = r_distance + r_orientation * 0.001
    return r_sum


def compute_position_diff_scalar(distance_diff):
    r_diff = [-1.1, -2., 1.]
    idx_diff = (np.sign(distance_diff) + 1).astype(int)
    if isinstance(distance_diff, (list, np.ndarray)):
        orig_shape = idx_diff.shape
        s_distance = np.array(list(map(lambda x: r_diff[x],
                                       idx_diff.flatten())))
        s_distance = np.reshape(s_distance, orig_shape)
    else:
        s_distance = r_diff[idx_diff]
    return s_distance


def compute_distance_reward(distance, distance_threshold=25.,
                            threshold_offset=5., n_segments=3):
    curr_distance = distance
    safety_distance = distance_threshold - threshold_offset / 2.
    r_distance = 1 - abs(1 - curr_distance / safety_distance)
    # create a distance scale intervals
    if curr_distance < safety_distance:
        r_distance *= 0.5
    for n in range(n_segments):
        if curr_distance < distance_threshold + threshold_offset * n * 1.5:
            r_distance *= 2.

    # normalize
    r_distance /= 2**n_segments

    if curr_distance < distance_threshold - threshold_offset:
        r_distance -= 1.

    r_distance = max(-1., r_distance)
    return r_distance


def compute_target_distance_reward(ref_position, pos_t, pos_t1,
                                   distance_threshold=25., threshold_offset=5.,
                                   n_segments=3):
    # compute central difference
    d1 = compute_distance(ref_position, pos_t)
    d2 = compute_distance(ref_position, pos_t1)
    reward = compute_distance_reward(d2, distance_threshold=distance_threshold,
                                     threshold_offset=threshold_offset,
                                     n_segments=n_segments)
    # reward = reward * compute_position_diff_scalar(d1 - d2)
    return reward


def compute_position2target_reward(ref_position, pos_t, pos_t1, orientation_t1,
                                   distance_threshold=25., disstance_offset=5.,
                                   orientation_offset=0.1, n_segments=3):
    # r_orientation = compute_target_orientation_reward(
    #     ref_position, pos_t1, orientation_t1,
    #     threshold_offset=orientation_offset, n_segments=n_segments)
    r_distance = compute_target_distance_reward(
        ref_position, pos_t, pos_t1, distance_threshold=distance_threshold,
        threshold_offset=disstance_offset, n_segments=n_segments)
    # r_sum = sum_and_normalize(r_distance, r_orientation)
    return r_distance

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def calculate_reward_grid(position_range, ref_position, ref_orientation):
        # Create a grid of x, y coordinates
        x_values = np.linspace(position_range[0], position_range[1], num=100)
        y_values = np.linspace(position_range[1], position_range[0], num=100)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        # Initialize grid to store reward values
        distance_grid = np.zeros((len(x_values), len(y_values), 2))
        orientation_grid = np.zeros((len(x_values), len(y_values)))

        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                # Calculate distance reward
                pos_distance = compute_distance(ref_position, [x, y])
                distance_grid[i, j, 0] = pos_distance
                distance_grid[i, j, 1] = compute_distance_reward(
                    pos_distance, distance_threshold=36.5, threshold_offset=5.)
                # Calculate orientation reward
                orientation_grid[i, j] = compute_orientation_reward(
                    ref_position, [x, y], ref_orientation)

        return x_grid, y_grid, orientation_grid, distance_grid

    def calculate_distance_derivative(x_grid, y_grid, reward_grid):
        dx = x_grid[0, 1] - x_grid[0, 0]  # Step size in x direction
        dy = y_grid[1, 0] - y_grid[0, 0]  # Step size in y direction

        # Compute the derivatives using central differences
        d_reward_dx_right = np.gradient(reward_grid, axis=1) / dx
        d_reward_dx_left = -d_reward_dx_right  # Reverse the direction for left derivative
        d_reward_dy_up = np.gradient(reward_grid, axis=0) / dy
        d_reward_dy_down = -d_reward_dy_up  # Reverse the direction for down derivative

        return (d_reward_dx_right, d_reward_dx_left,
                d_reward_dy_up, d_reward_dy_down)

    def compute_total_derivatives(*args):
        # Normalize, power of 2, sum, and sqrt
        return np.sqrt(np.sum([((d + 1) /2)**2 for d in args], axis=0))

    def plot_reward_heatmap(x_grid, y_grid, reward_grid, title):
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x_grid, y_grid, reward_grid, cmap='viridis')
        plt.colorbar(label='Reward')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.title(title)
        plt.grid()
        plt.show()

    # Define the range of positions and orientations
    agent_position_range = (-50, 50)  # Example range for x and y coordinates
    plot_all = True

    # Define the target position
    target_position = [-40, 40]  # Example target position
    target_orientation = compute_orientation([0, 0], target_position)

    # Calculate the reward grid
    x_grid, y_grid, orientation_grid, distance_grid = \
        calculate_reward_grid(agent_position_range, target_position, target_orientation)

    r_orientation = orientation_grid
    r_position = distance_grid[:, :, 0]
    r_distance = distance_grid[:, :, 1]
    r_dist_ori = sum_and_normalize(r_distance, r_orientation)

    print('distance_rewards', r_distance.min(), r_distance.max())
    print('orientation_rewards', r_orientation.min(), r_orientation.max())
    print('r_dist_ori', r_dist_ori.min(), r_dist_ori.max())

    # Plot the reward heatmap
    if plot_all:
        plot_reward_heatmap(x_grid, y_grid, r_distance,
                            'Position rewards')
        plot_reward_heatmap(x_grid, y_grid, r_orientation,
                            f'Orientation {target_orientation:.4f} rad rewards')
        plot_reward_heatmap(x_grid, y_grid, r_dist_ori,
                            'Position and orientation rewards')

    def plot_derivatives(derivatives, directions=['up', 'down', 'left', 'right'],
                         plot_all=False, norm=True):
        total_derivatives = list()
        dir_idx = ['right', 'left', 'up', 'down']
        for d in directions:
            d_idx = dir_idx.index(d)
            total_derivatives.append(derivatives[d_idx])
            if plot_all:
                plot_reward_heatmap(x_grid, y_grid, derivatives[d_idx],
                                    f"Position derivative {d} direction")
        # sum
        s_dir = '-'.join(directions)
        dist_derivatives_total = compute_total_derivatives(*total_derivatives)
        plot_reward_heatmap(x_grid, y_grid, dist_derivatives_total,
                            f"Position derivative {s_dir} direction")
        if norm:
            dist_derivatives_norm = min_max_norm(
                dist_derivatives_total, -1, 1,
                dist_derivatives_total.min(), dist_derivatives_total.max())
        
            plot_reward_heatmap(x_grid, y_grid, dist_derivatives_norm,
                                f"Position derivative norm. {s_dir} direction")

    distance_threshold = 40
    threshold_offset = 5
    safety_distance = distance_threshold - threshold_offset / 2
    better_pos_limits = (distance_threshold - threshold_offset,
                         distance_threshold)

    plot_reward_heatmap(x_grid, y_grid, r_position, 'Position reward')
    dist_derivatives = calculate_distance_derivative(x_grid, y_grid, r_position)
    plot_derivatives(dist_derivatives, ['up', 'left'])

    r_position2 = 1 - (r_position / safety_distance)
    plot_reward_heatmap(x_grid, y_grid, r_position2, 'Position reward 2')
    dist_derivatives2 = calculate_distance_derivative(x_grid, y_grid, r_position2)
    plot_derivatives(dist_derivatives2, ['up', 'left'])

    r_position3 = 1 - abs(r_position2)
    plot_reward_heatmap(x_grid, y_grid, r_position3, 'Position reward 3')
    dist_derivatives2 = calculate_distance_derivative(x_grid, y_grid, r_position3)
    plot_derivatives(dist_derivatives2, ['up', 'left'])

    r_position4 = r_position3.copy()
    r_position4[r_position < safety_distance] *= 0.5
    r_position4[r_position < distance_threshold - threshold_offset] -= 1.
    n_segments = 3
    for n in range(n_segments + 1):
        r_position4[r_position < distance_threshold +
                    threshold_offset * n * 1.2] *= 2.

    r_position4[r_position4 < -1.] = -1.
    plot_reward_heatmap(x_grid, y_grid, r_position4, 'Position reward 4')

    dist_derivatives2 = calculate_distance_derivative(x_grid, y_grid, r_position4)
    plot_derivatives(dist_derivatives2, ['up', 'left'])

    r_position5 = r_position.copy()
    orig_shape = r_position5.shape
    r_position5 = np.array(list(map(lambda x: compute_distance_reward(
        x, distance_threshold=40., threshold_offset=5., n_segments=3),
        r_position5.flatten())))
    r_position5 = np.reshape(r_position5, orig_shape)
    plot_reward_heatmap(x_grid, y_grid, r_position5, 'Position reward 5')

    dist_derivatives2 = calculate_distance_derivative(x_grid, y_grid, r_position5)
    plot_derivatives(dist_derivatives2, ['up', 'left'])
