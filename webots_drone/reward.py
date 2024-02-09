#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:35:08 2023

@author: Angel Ayala
"""

import numpy as np
from webots_drone.utils import compute_distance
from webots_drone.utils import compute_orientation


def compute_direction_vector(position, orientation, ref_position):
    # Calculate the vector pointing from the agent's position to the target position
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


def sum_and_normalize(orientation_rewards, distance_rewards, distance_diff=None):
    r_distance = (distance_rewards + 1.)
    r_orientation = (orientation_rewards + 1.)
    r_sum = r_distance * r_orientation
    if distance_diff is not None:
        r_sum *= np.sign(distance_diff)
    return r_sum - 1.


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
                distance_grid[i, j, 0] = compute_distance([x, y], ref_position)
                distance_grid[i, j, 1] = compute_distance_reward(
                    ref_position, [x, y],
                    distance_max=50., distance_threshold=36.5,
                    threshold_offset=5.)
                # Calculate orientation reward
                orientation_grid[i, j] = compute_orientation_reward(
                    [x, y], ref_orientation, ref_position)

        return x_grid, y_grid, orientation_grid, distance_grid

    def calculate_reward_derivative(x_grid, y_grid, reward_grid):
        dx = x_grid[0, 1] - x_grid[0, 0]  # Step size in x direction
        dy = y_grid[0, 0] - y_grid[1, 0]  # Step size in y direction

        # Compute the derivatives using central differences
        d_reward_dx = np.gradient(reward_grid, axis=0) / dx
        d_reward_dy = np.gradient(reward_grid, axis=1) / dy

        # Normalize
        d_reward_dx_norm = (d_reward_dx + 1) / 2.
        d_reward_dy_norm = (d_reward_dy + 1) / 2.

        # Compute total derivatives
        total_derivative = np.sqrt(d_reward_dx_norm**2 + d_reward_dy_norm**2)

        return total_derivative

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
    r_dist_ori = sum_and_normalize(r_orientation, r_distance)

    print('distance_rewards', r_distance.min(), r_distance.max())
    print('orientation_rewards', r_orientation.min(), r_orientation.max())
    print('r_dist_ori', r_dist_ori.min(), r_dist_ori.max())

    # Plot the reward heatmap
    if plot_all:
        plot_reward_heatmap(x_grid, y_grid, r_dist_ori,
                            'Position and orientation rewards')
        plot_reward_heatmap(x_grid, y_grid, r_orientation,
                            f'Orientation {target_orientation:.4f} rad rewards')
        plot_reward_heatmap(x_grid, y_grid, r_distance,
                            'Position rewards')

    # Plot the direction derivative scalar
    distance_d = calculate_reward_derivative(x_grid, y_grid,
                                             distance_grid[:, :, 0])
    print('distance_d', distance_d.min(), distance_d.max())

    if plot_all:
        plot_reward_heatmap(x_grid, y_grid, distance_d,
                     'Distance derivative of X and Y rewards')

    r_dist_ori_d = sum_and_normalize(r_orientation, r_distance,
                                     distance_diff=distance_d)
    print('r_dist_ori_d', r_dist_ori_d.min(), r_dist_ori_d.max())
    plot_reward_heatmap(x_grid, y_grid, r_dist_ori_d,
                        'Position, orientation, and distance derivatives rewards')
