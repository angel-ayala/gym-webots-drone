#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:35:08 2023

@author: Angel Ayala
"""

import numpy as np
from webots_drone.utils import compute_distance
from webots_drone.utils import compute_target_orientation
from webots_drone.utils import check_target_distance
from webots_drone.utils import check_same_position
from webots_drone.utils import min_max_norm
from webots_drone.utils import target_mask


def orientation2reward(orientation, ref_orientation):
    return np.cos(ref_orientation - orientation)


def distance2reward(distance, ref_distance):
    return 1. - abs(1. - distance / ref_distance)


def velocity2reward(velocity, pos_thr=0.003):
    dist_diff = velocity
    dist_diff *= np.abs(velocity).round(3) > pos_thr  # ensure minimum diff
    # if dist_diff > 0.:
    #     dist_diff *= 2.
    return dist_diff / 0.03


def compute_vector_reward(ref_position, pos_t, pos_t1, orientation_t1,
                          distance_target=36., distance_margin=5.):
    # compute orientation reward
    ref_orientation = compute_target_orientation(pos_t1, ref_position)
    r_orientation = orientation2reward(orientation_t1, ref_orientation)
    # compute distance reward
    ref_distance = distance_target - distance_margin / 2.
    dist_t1 = compute_distance(pos_t1, ref_position)
    r_distance = distance2reward(dist_t1, ref_distance)
    # compute velocity reward
    dist_t = compute_distance(pos_t, ref_position)
    r_velocity = velocity2reward(dist_t - dist_t1)
    # check zones
    zones = check_target_distance(dist_t1, distance_target,
                                  distance_margin / 2.)
    # inverse when trespass risk distance
    if zones[0]:
        r_velocity *= -1

    # compose reward
    # r_sum = r_distance + r_velocity + r_orientation
    r_sum = r_distance * 0.2 + r_velocity * 0.4 + r_orientation * 0.4

    # bonus in-distance
    if zones[1]:
        r_sum += 2.
    # penalty no movement
    if check_same_position(pos_t, pos_t1):
        r_sum -= 2.

    return r_sum


def compute_visual_reward(observation):
    reward = 0
    # channel first -> last
    observation = np.transpose(observation, axes=(1, 2, 0))
    obs_shape = observation.shape
    tmask, tarea, tcenter = target_mask(observation)
    if tcenter is not None:
        offset_x = abs(tcenter[0] - obs_shape[0] // 2)
        offset_y = abs(tcenter[1] - obs_shape[1] // 2)
        reward = tarea - (offset_x + offset_y)
        reward /= 800  # empirical desired area
    return reward


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
                position = [x, y]
                # Calculate distance reward
                distance = compute_distance(position, ref_position)
                distance_grid[i, j, 0] = distance
                distance_grid[i, j, 1] = distance2reward(distance, ref_distance=36.5)
                # Calculate orientation reward
                orientation = compute_target_orientation(position, ref_position)
                orientation_grid[i, j] = orientation2reward(orientation, ref_orientation)

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

    def apply_fn_list(array, fn):
        orig_shape = array.shape
        new_array = np.array(list(map(fn, array.flatten())))
        new_array = np.reshape(new_array, orig_shape)
        return new_array

    # Define the range of positions and orientations
    agent_position_range = (-100, 100)  # Example range for x and y coordinates
    plot_all = True
    distance_target = 36.5
    distance_margin = 5
    d_central = distance_target - distance_margin / 2

    # Define the target position
    target_position = [-50, 50]
    initial_position = [0, 0]
    target_orientation = compute_target_orientation(initial_position, target_position)

    # Calculate the reward grid
    x_grid, y_grid, orientation_grid, distance_grid = \
        calculate_reward_grid(agent_position_range, target_position, target_orientation)

    r_orientation = orientation_grid
    p_distance = distance_grid[:, :, 0]
    p_distance2 = -(p_distance - d_central) / d_central
    r_distance = distance_grid[:, :, 1]
    r_dist_ori = r_distance * (r_orientation + 1.) / 2. - 1.

    print('distance_rewards', r_distance.min(), r_distance.max())
    print('orientation_rewards', r_orientation.min(), r_orientation.max())
    print('r_dist_ori', r_dist_ori.min(), r_dist_ori.max())

    # Plot the reward heatmap
    plot_reward_heatmap(x_grid, y_grid, p_distance2, 'Position distances, target centered')
    plot_reward_heatmap(x_grid, y_grid, r_distance, 'Distance rewards')
    plot_reward_heatmap(x_grid, y_grid, r_orientation, f'Orientation {target_orientation:.4f} rad rewards')
    plot_reward_heatmap(x_grid, y_grid, r_dist_ori, 'Distance and orientation rewards')

    # Plot derivatives
    # distance
    # d_distance = calculate_distance_derivative(x_grid, y_grid, r_distance, ['up', 'left'])
    # plot_reward_heatmap(x_grid, y_grid, d_distance, 'Distance reward derivatives')
    # orientation
    # d_orientation = calculate_distance_derivative(x_grid, y_grid, r_orientation, ['up', 'left'])
    # plot_reward_heatmap(x_grid, y_grid, d_orientation, 'Orientation reward derivatives')
    # distance + orientation
    d_dist_ori = calculate_distance_derivative(x_grid, y_grid, r_dist_ori, ['up', 'left'])
    plot_reward_heatmap(x_grid, y_grid, d_dist_ori, 'Distance and orientation reward derivatives')

    # target approximation
    # Going away (pos diff)
    d_pos_down = calculate_distance_derivative(x_grid, y_grid, p_distance2, ['up', 'left'], norm=False)
    down_factor = d_pos_down * r_orientation - 1
    plot_reward_heatmap(x_grid, y_grid, down_factor, 'Velocity reward (oppposite to target)')
    # plot_reward_heatmap(x_grid, y_grid, r_dist_ori - down_factor, 'Distance and orientation - down_factor')
    # Aproaching (neg diff)
    d_pos_up = -d_pos_down
    up_factor = d_pos_up * r_orientation - 1
    plot_reward_heatmap(x_grid, y_grid, up_factor, 'Velocity reward (towards target)')
    # plot_reward_heatmap(x_grid, y_grid, r_dist_ori + up_factor, 'Distance and orientation + up_factor')
    # No diff
    # no_factor = apply_fn_list(np.zeros_like(p_distance2), compute_velocity_reward)
    # plot_reward_heatmap(x_grid, y_grid, no_factor, 'No position difference')
    # plot_reward_heatmap(x_grid, y_grid, r_dist_ori + no_factor, 'Distance and orientation + no_factor')
