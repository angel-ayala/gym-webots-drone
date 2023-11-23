#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:35:08 2023

@author: Angel Ayala
"""

import numpy as np
from webots_drone.utils import compute_distance
from webots_drone.utils import min_max_norm


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

    return cosine_similarity


def compute_distance_reward(position, ref_position, distance_max=25.,
                            distance_threshold=5.,
                            threshold_offset=2.):
    curr_distance = compute_distance(position, ref_position)

    if curr_distance < distance_threshold - threshold_offset:
        return -1.
    if curr_distance < distance_threshold:
        return 1.

    return max(-1., 1. - curr_distance / distance_max)


def sum_and_normalize(orientation_rewards, distance_rewards):
    sum_rewards = (1. + orientation_rewards) * (1. + distance_rewards)
    sum_rewards /= 4.
    return sum_rewards


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def calculate_reward_grid(position_range, orientation_range, ref_position):
        # Create a grid of x, y coordinates
        x_values = np.linspace(*position_range, num=100)
        y_values = np.linspace(*position_range, num=100)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        # Create a grid for orientation values (assuming radians)
        orientation_values = np.linspace(*orientation_range, num=100)

        # Initialize grid to store reward values
        distance_grid = np.zeros((len(x_values), len(y_values)))
        orientation_grid = np.zeros((len(x_values), len(y_values),
                                     len(orientation_values)))

        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                distance_reward = compute_distance_reward([x, y], ref_position)
                distance_grid[i, j] += distance_reward
                for k, orientation in enumerate(orientation_values):
                    # Calculate reward for each combination
                    orientation_reward = compute_orientation_reward(
                        [x, y], orientation, ref_position)
                    orientation_grid[i, j, k] += orientation_reward

        return x_grid, y_grid, orientation_grid, distance_grid

    def plot_reward_heatmap(x_grid, y_grid, reward_grid):
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x_grid, y_grid, reward_grid, cmap='viridis')
        plt.colorbar(label='Reward')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.title('Reward Heatmap considering Position (X, Y) and Orientation')
        plt.show()

    # Define the range of positions and orientations
    agent_position_range = (-100, 100)  # Example range for x and y coordinates
    orientation_range = (0, 2 * np.pi)  # Example range for orientation (radians)

    # Define the target position
    target_position = [40, -40]  # Example target position

    # Calculate the reward grid
    x_grid, y_grid, orientation_grid, distance_grid = \
        calculate_reward_grid(agent_position_range, orientation_range, target_position)

    orientation_rewards = orientation_grid.sum(axis=2)
    distance_rewards = distance_grid
    print('distance_rewards',
          distance_rewards.min(), distance_rewards.max())
    print('orientation_rewards',
          orientation_rewards.min(), orientation_rewards.max())
    total_rewards = sum_and_normalize(orientation_rewards, distance_rewards)

    # Plot the reward heatmap
    plot_reward_heatmap(x_grid, y_grid, orientation_rewards)
    plot_reward_heatmap(x_grid, y_grid, distance_rewards)
    plot_reward_heatmap(x_grid, y_grid, total_rewards)
