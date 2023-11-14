#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:11:59 2023

@author: Angel Ayala
"""
import json
import pandas as pd
from webots_drone.utils import info2state


class SetEpisode:
    def __init__(self):
        print("Setting episodes info")
        self.last_ep = 1

    def __call__(self, last):
        set_ep = self.last_ep
        self.last_ep += int(last)
        return set_ep


class SetPhase:
    def __init__(self, learn_steps, eval_steps, mem_steps):
        self.current = 'fillmem'
        self.ep = 0
        self.t = 1
        self.n = 0
        self.n_mem = mem_steps
        self.n_eval = eval_steps
        self.n_learn = learn_steps
        self.ep_end = self.n_learn + self.n_eval
        # self.counter = dict(fillmem=0, learn=0, eval=0)

    def _print(self, extra=None):
        print(self.n,
              (self.n - self.n_mem),
              self.t, self.current,
              extra, self.counter)

    # def reset(self):
    #     self.ep = 0
    #     self.t = 1
    #     self.n = 0

    def __call__(self, last):
        if self.n > 0:
            if self.n == self.n_mem and self.current == 'fillmem':
                self.current = 'filleval'

            if self.n == (self.n_eval * self.t
                          + self.n_learn * (self.t - 1)
                          + self.n_mem):
                self.current = 'learn'
                self.ep += 1

            if self.n == (self.ep_end * self.t + self.n_mem):
                self.current = 'eval'
                self.t += 1

        self.n += 1
        # self.counter[self.current] += 1

        return self.current, self.ep

    # def get_ep(self, last):
    #     self(last)
    #     return self.ep


def read_args(args_path):
    print('Reading arguments from:', args_path)
    args = dict()
    with open(args_path, 'r') as f:
        args = json.load(f)
    return args


def read_history(history_path, is_full_history=False):
    print('Reading history file from:', history_path)
    history_df = pd.read_csv(history_path)
    # preprocess data
    ep_set = SetEpisode()
    history_df['ep'] = history_df['last'].apply(ep_set)
    if is_full_history:
        args_path = history_path.parent / 'args_train.json'
        apply_phase(history_df, args_path)

    return history_df


def apply_phase(history_df, args_path):
    exp_args = read_args(args_path)
    print("Setting episodes phase")
    phase_set = SetPhase(exp_args['steps_train'],
                         exp_args['steps_eval'],
                         exp_args['init_replay_size'])
    history_df[['phase', 'alg_ep']] = history_df['last'].apply(phase_set)
    return history_df


class StoreStepData:
    """Callback for save a Gym.state data."""

    def __init__(self, store_path, is_obs_img=True):
        self.is_obs_img = is_obs_img
        self.store_path = store_path
        state_cols = ['pos_x', 'pos_y', 'pos_z',
                      'ori_x', 'ori_y', 'ori_z',
                      'vel_x', 'vel_y', 'vel_z',
                      'velang_x', 'velang_y', 'velang_z',]
        action_cols = ['roll', 'pitch', 'yaw', 'thrust']

        self.data_cols = ['timestamp'] + state_cols
        self.data_cols += action_cols
        self.data_cols += ['reward']
        self.data_cols += state_cols
        self.data_cols += ['absorbing', 'last']
        self._idx = 0
        self.last_state = None

    def __call__(self, observation, info):
        # format state data
        sample = observation[0]
        state = info2state(info).tolist()

        row = list()
        row.append(info['timestamp'])  # action
        row.extend(self.last_state)  # state
        row.extend(sample[1])  # action
        row.append(sample[2])  # reward
        row.extend(state)  # next state
        row.append(sample[4])  # absorbing
        row.append(sample[5])  # last

        for k, v in info.items():
            if k in ['position', 'orientation', 'angular_velocity',
                     'image', 'angular_vel',
                     'timestamp', 'motors_vel',
                     'emitter', 'rc_position']:
                continue

            if isinstance(v, list):
                if k not in self.data_cols:
                    for i in range(len(v)):
                        self.data_cols.append(k + f"_{i}")
                row.extend(v)
            else:
                if k not in self.data_cols:
                    self.data_cols.append(k)
                row.append(v)

        k = 'rc_position'
        self.data_cols.extend([k + '_x', k + '_y', k + '_z'])
        row.extend(info[k])

        k = 'emitter'
        self.data_cols.extend([k + '_pos_x', k + '_pos_y', k + '_pos_z',
                               k + '_strength'])
        row.extend(info[k]['direction'])
        row.append(info[k]['signal_strength'])

        k = 'motors_vel'
        self.data_cols.extend([k + '_front_left',
                               k + '_front_right',
                               k + '_rear_left',
                               k + '_rear_right'])
        row.extend(info[k])

        if self._idx == 0:  # create output file
            with open(self.store_path, 'w') as outfile:
                outfile.writelines(','.join(self.data_cols) + '\n')

        with open(self.store_path, 'a') as outfile:
            outfile.writelines(','.join(map(str, row)) + '\n')

        self._idx += 1