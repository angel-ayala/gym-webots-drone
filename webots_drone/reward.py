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
                                      threshold_offset=0.025, n_segments=5):
    curr_distance = compute_orientation_reward(ref_position, position,
                                               orientation)
    r_distance = np.copy(curr_distance)
    r_distance = add_scales(r_distance, curr_distance, 1,
                            thr_offset=-threshold_offset, n_segments=n_segments,
                            direction=1)

    if curr_distance >= 1 - threshold_offset:
        r_distance *= 2.
    return r_distance


def apply_fn_list(array, fn):
    orig_shape = array.shape
    new_array = np.array(list(map(fn, array.flatten())))
    new_array = np.reshape(new_array, orig_shape)
    return new_array


def compute_distance_diff_scalar(distance_diff, diff_thr=0.003):
    # benefits distance reduction
    # r_diff = [1., -2., -0.1]
    # r_diff = [10., -5., -1.]
    # r_diff = [1., 0.001, 0.01]  # keeps doing circles
    # r_diff = [2., -2., -1.]
    # r_diff = [1., 0., 0.01]  # for testing
    # r_diff = [1., -2., -0.01]
    # r_diff = [1., -1., -0.1]
    distance_diff = np.round(distance_diff, 3)
    distance_diff *= np.abs(distance_diff) > diff_thr
    # idx_diff = (np.sign(distance_diff) + 1).astype(int)
    # if isinstance(distance_diff, (list, np.ndarray)):
    #     s_distance = apply_fn_list(idx_diff, lambda x: r_diff[x])
    # else:
    #     s_distance = r_diff[idx_diff]
    # return s_distance
    return distance_diff


def compare_direction(x, y, direction):
    if direction == 1:
        return x >= y
    if direction == 0:
        return x < y

def add_scales(reward, distance, central_point, thr_offset, n_segments=5,
               thr_factor=1.5, direction=0):
    scaled_reward = np.copy(reward)
    compare = lambda x, y: compare_direction(x, y, direction)
    if isinstance(distance, (list, np.ndarray)):
        scaled_reward[compare(distance, central_point)] *= 0.5
    elif compare(distance, central_point):
        scaled_reward *= 0.5

    for n in range(n_segments):
        if isinstance(distance, (list, np.ndarray)):
            scaled_reward[compare(distance, central_point + thr_offset * n * thr_factor)] *= 2.
        elif compare(distance, central_point + thr_offset * n * thr_factor):
            scaled_reward *= 2.

    # normalize
    scaled_reward /= 2**n_segments
    return scaled_reward


def compute_distance_reward(distance, d_central=None, distance_threshold=25.,
                            threshold_offset=5., n_segments=5):
    if d_central is None:
        d_central = distance_threshold - threshold_offset / 2.
    curr_distance = distance
    r_distance = 1. - abs(1. - curr_distance / d_central)
    r_distance = add_scales(r_distance, curr_distance, d_central,
                            thr_offset=threshold_offset, n_segments=n_segments)
    if distance_threshold > curr_distance > distance_threshold - threshold_offset:
        r_distance *= 2.

    return r_distance


def normalize_and_mul(distance_rewards, orientation_rewards):
    r_distance = (distance_rewards + 1) / 2.
    r_orientation = (orientation_rewards + 1) / 2.
    r_sum = distance_rewards + r_orientation * r_distance
    return r_sum


def compute_position2target_reward(ref_position, pos_t, pos_t1, orientation_t1,
                                   distance_threshold=36., distance_offset=5.,
                                   orientation_offset=0.015, n_segments=3):
    # compute orientation reward
    r_orientation = compute_target_orientation_reward(
        ref_position, pos_t1, orientation_t1,
        threshold_offset=orientation_offset, n_segments=n_segments)
    # compute distance reward
    dist_t1 = compute_distance(ref_position, pos_t1)
    d_central = distance_threshold - distance_offset / 2.
    r_distance = compute_distance_reward(
        dist_t1, d_central, distance_threshold=distance_threshold,
        threshold_offset=distance_offset, n_segments=n_segments)
    # normalize and sum
    r_sum = normalize_and_mul(r_distance, r_orientation)
    # compute distance momentum factor
    dist_t = abs(compute_distance(ref_position, pos_t) - d_central)
    r_sum *= compute_distance_diff_scalar(dist_t - abs(dist_t1 - d_central))

    # penalty off-distance
    if dist_t1 < distance_threshold - distance_offset:
        r_sum -= 25.
    # bonus in-distance
    elif dist_t1 < distance_threshold:
        r_sum += 50.

    return r_sum


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
                orientation_grid[i, j] = compute_target_orientation_reward(
                    ref_position, [x, y], ref_orientation)

        return x_grid, y_grid, orientation_grid, distance_grid

    def calculate_distance_derivative(x_grid, y_grid,reward_grid,
                                      directions=['up', 'down',
                                                  'left', 'right'], norm=True):
        dx = x_grid[0, 1] - x_grid[0, 0]  # Step size in x direction
        dy = y_grid[1, 0] - y_grid[0, 0]  # Step size in y direction

        # Compute the derivatives using central differences
        total_derivatives = list()
        d_reward_dx_right = np.gradient(reward_grid, axis=1) / dx
        d_reward_dy_up = np.gradient(reward_grid, axis=0) / dy

        if 'right' in directions:
            total_derivatives.append(((d_reward_dx_right + 1) / 2)**2)
        if 'left' in directions:
            # Reverse the direction for left derivative
            total_derivatives.append(((-d_reward_dx_right + 1) / 2)**2)
        if 'up' in directions:
            total_derivatives.append(((d_reward_dy_up + 1) / 2)**2)
        if 'down' in directions:
            # Reverse the direction for down derivative
            total_derivatives.append(((-d_reward_dy_up + 1) / 2)**2)

        # sum
        d_total = np.sqrt(np.sum(total_derivatives, axis=0))
        if norm:
            d_total = min_max_norm(d_total, -1, 1,
                                   d_total.min(), d_total.max())
        return d_total

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
    agent_position_range = (-100, 100)  # Example range for x and y coordinates
    plot_all = True
    distance_threshold = 36.5
    threshold_offset = 5
    d_central = distance_threshold - threshold_offset / 2
    better_pos_limits = (distance_threshold - threshold_offset,
                         distance_threshold)

    # Define the target position
    target_position = [-50, 50]  # Example target position
    target_orientation = compute_orientation([0, 0], target_position)

    # Calculate the reward grid
    x_grid, y_grid, orientation_grid, distance_grid = \
        calculate_reward_grid(agent_position_range, target_position, target_orientation)

    r_orientation = orientation_grid
    p_distance = distance_grid[:, :, 0]
    p_distance2 = -(p_distance - d_central)/ d_central
    r_distance = distance_grid[:, :, 1]
    r_dist_ori = normalize_and_mul(r_distance, r_orientation)

    print('distance_rewards', r_distance.min(), r_distance.max())
    print('orientation_rewards', r_orientation.min(), r_orientation.max())
    print('r_dist_ori', r_dist_ori.min(), r_dist_ori.max())

    # Plot the reward heatmap
    plot_reward_heatmap(x_grid, y_grid, p_distance2, 'Position distances, target centered')
    plot_reward_heatmap(x_grid, y_grid, r_distance, 'Distance rewards')
    plot_reward_heatmap(x_grid, y_grid, r_orientation,
                        f'Orientation {target_orientation:.4f} rad rewards')
    plot_reward_heatmap(x_grid, y_grid, r_dist_ori,
                        'Distance and orientation rewards')
    # Plot derivatives
    d_dist_ori = calculate_distance_derivative(x_grid, y_grid, r_dist_ori, ['up', 'left'])
    plot_reward_heatmap(x_grid, y_grid, d_dist_ori, 'Distance and orientation reward derivatives')
    # distance
    d_distance = calculate_distance_derivative(x_grid, y_grid, r_distance, ['up', 'left'])
    plot_reward_heatmap(x_grid, y_grid, d_distance, 'Distance reward derivatives')
    # orientation
    d_orientation = calculate_distance_derivative(x_grid, y_grid, r_orientation, ['up', 'left'])
    plot_reward_heatmap(x_grid, y_grid, d_orientation, 'Orientation reward derivatives')
    # distance + orientation
    d_dist_ori = normalize_and_mul(d_distance, d_orientation)
    plot_reward_heatmap(x_grid, y_grid, d_dist_ori, 'Distance and orientation rewards derivatives')

    # target approximation
    p_distance2 = np.abs(p_distance - d_central)
    plot_reward_heatmap(x_grid, y_grid, p_distance2, 'Absolute centered position')
    # Aproachin (neg diff)
    d_pos_up = calculate_distance_derivative(x_grid, y_grid, p_distance2, ['up', 'left'], norm=False)
    plot_reward_heatmap(x_grid, y_grid, d_pos_up, 'Negative position difference')
    up_factor = compute_distance_diff_scalar(d_pos_up)
    plot_reward_heatmap(x_grid, y_grid, up_factor, 'Negative position difference')
    plot_reward_heatmap(x_grid, y_grid, r_dist_ori * up_factor, 'Distance and orientation times up_factor')
    # Distance (pos diff)
    d_pos_down = calculate_distance_derivative(x_grid, y_grid, p_distance2, ['down', 'right'], norm=False)
    down_factor = compute_distance_diff_scalar(d_pos_down)
    plot_reward_heatmap(x_grid, y_grid, down_factor, 'Positive position difference')
    plot_reward_heatmap(x_grid, y_grid, r_dist_ori * down_factor, 'Distance and orientation times down_factor')
    # No diff
    no_factor = compute_distance_diff_scalar(np.zeros_like(d_pos_down))
    plot_reward_heatmap(x_grid, y_grid, no_factor, 'No position difference')
    plot_reward_heatmap(x_grid, y_grid, r_dist_ori * no_factor, 'Distance and orientation times no_factor')

    # compose = np.abs(r_dist_ori) * down_factor + np.abs(r_dist_ori) * up_factor
    compose = r_dist_ori * up_factor + r_dist_ori * down_factor + r_dist_ori * no_factor
    plot_reward_heatmap(x_grid, y_grid, compose, 'Distance and orientation times (up_ + down_ + no_factor)')
    d_compose = calculate_distance_derivative(x_grid, y_grid, compose, ['up', 'left'], norm=False)
    plot_reward_heatmap(x_grid, y_grid, d_compose, 'Distance and orientation times (up_ + down_ + no_factor) derivative pos')
    d_compose = calculate_distance_derivative(x_grid, y_grid, compose, ['down', 'right'], norm=False)
    plot_reward_heatmap(x_grid, y_grid, d_compose, 'Distance and orientation times (up_ + down_ + no_factor) derivative neg')

