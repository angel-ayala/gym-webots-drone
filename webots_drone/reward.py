#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:35:08 2023

@author: Angel Ayala
"""

import numpy as np
from webots_drone.utils import compute_distance


def compute_vector(position, ref_position):
    # Calculate the vector pointing from the agent's position to the target position
    vector_to_target = np.array(ref_position) - np.array(position)

    # Normalize the vector
    norm_vector_to_target = vector_to_target / np.linalg.norm(vector_to_target)
    return norm_vector_to_target


def compute_direction_vector(position, orientation, ref_position):
    norm_vector_to_target = compute_vector(position, ref_position)

    # Calculate the agent's forward vector based on its orientation
    agent_forward_vector = np.array([np.cos(orientation),
                                     np.sin(orientation)])

    return norm_vector_to_target, agent_forward_vector


def compute_orientation_reward(position, orientation, ref_position,
                               facing_threshold=0.995):
    # Get direction vectors
    direction_to_target, agent_forward = compute_direction_vector(
        position, orientation, ref_position)
    # print('direction_to_target', direction_to_target)

    # Calculate cosine similarity between the direction to the target and agent's forward direction
    cosine_similarity = np.dot(direction_to_target, agent_forward)
    # print(cosine_similarity)

    # Give rewards based on the agent's facing direction towards the target
    if cosine_similarity > facing_threshold:
        reward = 1.0  # Maximum reward when agent is facing the target
    else:
        reward = cosine_similarity

    return reward


def compute_distance_reward(position, ref_position):
    return 1.0 - compute_distance(position, ref_position)


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
        reward_grid = np.zeros((len(x_values), len(y_values),
                                len(orientation_values)))

        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                for k, orientation in enumerate(orientation_values):
                    # Calculate reward for each combination
                    reward = compute_orientation_reward(
                        [x, y], orientation, ref_position)
                    # reward = compute_distance_reward(
                    #     [x, y], ref_position)
                    reward_grid[i, j, k] += reward

        return x_grid, y_grid, reward_grid

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
    x_grid, y_grid, reward_grid = calculate_reward_grid(agent_position_range, orientation_range, target_position)

    # Plot the reward heatmap
    plot_reward_heatmap(x_grid, y_grid, reward_grid.mean(axis=2))
