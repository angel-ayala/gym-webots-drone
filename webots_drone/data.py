#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:11:59 2023

@author: Angel Ayala
"""
import json
import pandas as pd
from pathlib import Path
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

    def _print(self, extra=None):
        print(self.n,
              (self.n - self.n_mem),
              self.t, self.current,
              extra, self.counter)

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

        return self.current, self.ep


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


class MultipleCallbacksOnStep:
    def __init__(self):
        self._callbacks = list()

    def append(self, callback):
        self._callbacks.append(callback)

    def __call__(self, dataset, info):
        for c in self._callbacks:
            c(dataset, info)


class StoreStepData:
    """Callback for save a Gym.state data."""

    def __init__(self, store_path, epsilon=False):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if self.store_path.is_file():
            print('WARNING:', self.store_path, 'already exists, overwriting!')
            
        state_cols = ['pos_x', 'pos_y', 'pos_z',
                      'ori_x', 'ori_y', 'ori_z',
                      'vel_x', 'vel_y', 'vel_z',
                      'velang_x', 'velang_y', 'velang_z',]
        # action_cols = ['roll', 'pitch', 'yaw', 'thrust']
        action_cols = ['action']

        self.data_cols = ['phase', 'ep', 'iteration', 'timestamp']
        self.data_cols += state_cols
        self.data_cols += action_cols
        self.data_cols += ['reward']
        self.data_cols += state_cols
        self.data_cols += ['absorbing', 'last']
        self._idx = 0
        self.last_state = info2state(None).tolist()
        self._phase = 'init'
        self._ep = 0
        self._iteration = 0
        self.epsilon = epsilon
        if epsilon is not False:
            self.data_cols += ['epsilon']

    def set_learning(self):
        self._phase = 'learn'
        self._iteration = 0
        self._ep += 1

    def set_eval(self):
        self._phase = 'eval'
        self._iteration = 0
        self._ep += 1

    def __call__(self, sample, info):
        state = info2state(info).tolist()
        row = list()
        row.extend([self._phase, self._ep, self._iteration])  # epdata
        row.append(info['timestamp'])  # action
        row.extend(self.last_state)  # state
        # action
        if type(sample[1]) == list:
            row.extend(sample[1])
        else:
            row.append(sample[1])
        row.append(sample[2])  # reward
        row.extend(state)  # next state
        row.append(sample[4])  # absorbing
        row.append(sample[5])  # last
        if self.epsilon is not False:
            row.append(self.epsilon.get_value())  # epsilon

        for k, v in info.items():
            if k in ['position', 'orientation', 'angular_velocity', 'speed'
                     'image', 'angular_vel', 'timestamp', 'motors_vel',
                     'emitter', 'rc_position', 'target_position']:
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

        k = 'target_position'
        self.data_cols.extend([k + '_x', k + '_y', k + '_z'])
        row.extend(info[k])

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
        self.last_state = state

        if sample[5]:
            self._iteration += 1


class VideoCallback:
    def __init__(self, store_path, mdp):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.env_sim = mdp.env.sim
        self._recording = False
        self._iteration = 0

    def _start_recording(self):
        self.vid_path = self.store_path / f"iteration_{self._iteration:03d}.mp4"
        if self.vid_path.is_file():
            print('WARNING:', self.vid_path, 'already exists, overwriting!')
        if self.env_sim.movieIsReady():
            self.env_sim.movieStartRecording(str(self.vid_path.absolute()),
                                             width=848, height=480,
                                             quality=100, codec=0,
                                             acceleration=5, caption=True)
            if self.env_sim.movieFailed():
                print('Error on recording movie...')
            else:
                self._recording = True
        else:
            print('Record not started, previous encoding still running.')

    def _stop_recording(self):
        self.env_sim.movieStopRecording()
        self._recording = False

    def __call__(self, sample, info):
        if not self._recording:
            self._start_recording()
        if sample[5]:
            self._iteration += 1
            self._stop_recording()
