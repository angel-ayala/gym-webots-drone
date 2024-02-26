#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:35:08 2023

@author: Angel Ayala
"""

import numpy as np
from webots_drone.utils import compute_distance
from webots_drone.utils import compute_orientation


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


def compute_current_distance_reward(ref_position, position,
                                    distance_threshold=25.,
                                    threshold_offset=5.):
    curr_distance = compute_distance(position, ref_position)
    safety_distance = distance_threshold - threshold_offset / 2
    reward = 1 - abs(1 - curr_distance / safety_distance)
    reward = max(-1., reward)

    if curr_distance < distance_threshold - threshold_offset:
        return -1.

    return reward


def sum_and_normalize(distance_rewards, orientation_rewards, distance_diff=1.):
    r_distance = (distance_rewards + 1.) / 2.
    r_orientation = (orientation_rewards + 1.) / 2.
    r_distance_d = np.sign(distance_diff)
    r_sum = r_distance_d * 3. + r_distance * 2. + r_orientation * 0.1
    return r_sum


def compute_target_distance_reward(ref_position, pos_t, pos_t1,
                                   distance_threshold=25.,
                                   threshold_offset=5.):
    distance_t = compute_distance(ref_position, pos_t)
    distance_t1 = compute_distance(ref_position, pos_t1)

    if distance_t1 < distance_threshold - threshold_offset:
        return -100.

    if distance_t1 < distance_threshold:
        return 100.

    dist_vals = [-0.01, -1., 1.]
    r_distance = np.round(distance_t - distance_t1, 3)
    r_distance *= np.abs(r_distance) > 0.003  # prevent rotation diff
    r_distance = dist_vals[int(np.sign(r_distance) + 1)]
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
                distance_grid[i, j, 0] = compute_distance(ref_position, [x, y])
                distance_grid[i, j, 1] = compute_current_distance_reward(
                    ref_position, [x, y], distance_threshold=36.5,
                    threshold_offset=5.)
                # Calculate orientation reward
                orientation_grid[i, j] = compute_orientation_reward(
                    ref_position, [x, y], ref_orientation)

        return x_grid, y_grid, orientation_grid, distance_grid

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
