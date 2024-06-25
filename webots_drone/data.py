#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:11:59 2023

@author: Angel Ayala
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from webots_drone.envs.preprocessor import info2state
from webots_drone.envs.drone_discrete import DroneEnvDiscrete
from webots_drone.utils import compute_risk_distance


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

    def __init__(self, store_path, n_sensors=9, epsilon=False):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if self.store_path.is_file():
            print('WARNING:', self.store_path, 'already exists, overwriting!')
        self.n_sensors = n_sensors
        self.epsilon = epsilon
        self._phase = 'init'
        self._ep = 0
        self._iteration = -1
        self.create_header()

    def set_init_state(self, state, info):
        self.last_state = info2state(info).tolist()
        self._iteration += 1

    def new_episode(self):
        self._ep += 1
        self._iteration = -1

    def set_learning(self):
        self._phase = 'learn'

    def set_eval(self):
        self._phase = 'eval'

    def create_header(self):
        state_cols = ['pos_x', 'pos_y', 'pos_z',
                      'ori_x', 'ori_y', 'ori_z',
                      'vel_x', 'vel_y', 'vel_z',
                      'velang_x', 'velang_y', 'velang_z', 'north_rad']
        # action_cols = ['roll', 'pitch', 'yaw', 'thrust']
        action_cols = ['action']

        data_cols = ['phase', 'ep', 'iteration', 'timestamp']
        data_cols += state_cols
        data_cols += action_cols
        data_cols += ['reward']
        data_cols += ['next_' + sc for sc in state_cols]
        data_cols += ['absorbing', 'last']

        if self.epsilon is not False:
            data_cols += ['epsilon']
        data_cols += ['penalization', 'final']

        for nid in range(self.n_sensors):
            data_cols += ['dist_sensor_' + str(nid)]

        data_cols += ['target_pos_x',
                      'target_pos_y',
                      'target_pos_z']
        data_cols += ['target_dim_height',
                      'target_dim_radius']
        data_cols += ['rc_pos_x',
                      'rc_pos_y',
                      'rc_pos_z']
        data_cols += ['emitter_pos_x',
                      'emitter_pos_y',
                      'emitter_pos_z',
                      'emitter_strength']
        data_cols += ['motors_vel_front_left',
                      'motors_vel_front_right',
                      'motors_vel_rear_left',
                      'motors_vel_rear_right']
        # create file headers
        with open(self.store_path, 'w') as outfile:
            outfile.writelines(','.join(data_cols) + '\n')

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
            row.append(self.epsilon())  # epsilon

        row.append(info['penalization'])
        row.append(info['final'])

        for nid in range(self.n_sensors):
            row.append(info['dist_sensors'][nid])

        row.extend(info['target_position'])
        row.extend(info['target_dim'])
        row.extend(info['rc_position'])
        row.extend(info['emitter']['direction'])
        row.append(info['emitter']['signal_strength'])
        row.extend(info['motors_vel'])

        # append data
        with open(self.store_path, 'a') as outfile:
            outfile.writelines(','.join(map(str, row)) + '\n')

        self.last_state = state


class ExperimentData:
    def __init__(self, experiment_path, csv_name='history_training.csv',
                 env_args='args_environment.json',
                 train_args='args_training.json',
                 agent_args='args_agent.json'):
        if not isinstance(experiment_path, Path):
            experiment_path = Path(experiment_path)
        self.experiment_path = experiment_path
        print('Loading experiment data from:', self.experiment_path)
        self.history_df = pd.read_csv(self.experiment_path / csv_name)
        self.env_params = read_args(self.experiment_path / env_args)
        self.train_params = read_args(self.experiment_path / train_args)
        self.agent_params = read_args(self.experiment_path / agent_args)
        # append evaluation results
        self.join_eval_data()

    def join_eval_data(self, csv_regex=r'history_eval_*.csv'):
        csv_paths = list(self.experiment_path.rglob(csv_regex))
        csv_paths.sort()
        eval_df = None
        for csv_path in csv_paths:
            print('Loading evaluation data from:', csv_path)
            ep_df = pd.read_csv(csv_path)
            if eval_df is None:
                eval_df = ep_df.copy()
            else:
                eval_df = pd.concat((eval_df, ep_df))
        if eval_df is not None:
            self.history_df = pd.concat((self.history_df, eval_df))

    def get_reward_curve(self, phase='eval'):
        phase_df = self.history_df[self.history_df['phase'] == phase]
        reward_values = phase_df.groupby('ep').agg({'reward': 'sum'})
        return reward_values.to_numpy().flatten()

    def get_flight_area(self, world_name='forest_tower_200x200_simple'):
        import webots_drone

        with open(Path(webots_drone.__path__[0] +
                       f"/../worlds/{world_name}.wbt"), 'r') as world_dump:
            world_file = world_dump.read()

        area_size_regex = re.search(
            r"DEF FlightArea(.*\n)+  size( \-?\d+\.?\d?){2}\n",
            world_file)
        area_size_str = area_size_regex.group().strip()
        area_size = list(map(float, area_size_str.split(' ')[-2:]))
        # print('area_size', area_size)

        area_size = [fs / 2 for fs in area_size]  # size from center
        flight_area = [[fs * -1 for fs in area_size], area_size]

        # add altitude limits
        flight_area[0].append(self.env_params['altitude_limits'][0])
        flight_area[1].append(self.env_params['altitude_limits'][1])

        return flight_area

    def get_tuple_control(self, filtered_df):
        state_cols = ['pos_x', 'pos_y', 'pos_z',
                      'ori_x', 'ori_y', 'ori_z',
                      'vel_x', 'vel_y', 'vel_z',
                      'velang_x', 'velang_y', 'velang_z',
                      'north_rad']
        action_col = 'action'
        next_state_cols = ['next_' + sc for sc in state_cols]
        state_data = filtered_df[state_cols].to_numpy()
        next_state_data = filtered_df[next_state_cols].to_numpy()
        actions_data = np.zeros((len(filtered_df), 4))

        for i, (idx, row) in enumerate(filtered_df.iterrows()):
            if isinstance(row[action_col], int):
                actions_data[i] = DroneEnvDiscrete.discrete2continuous(
                    row[action_col])
            elif isinstance(row[action_col], str):
                numbers = row[action_col].split()
                if numbers[0] == '[':
                    numbers = numbers[1:]
                if numbers[-1] == ']':
                    numbers = numbers[:-1]
                daction = " ".join(numbers)
                daction = daction.replace('[', '').replace(']', '')
                if len(numbers) == 3:
                    daction += ' 0.'
                actions_data[i] = np.fromstring(daction, dtype=np.float32,
                                                sep=' ')
            else:
                actions_data[i] = row[action_col]

        return state_data, actions_data, next_state_data

    def get_phase_eps(self, phase):
        avbl_phase = self.history_df['phase'].unique().tolist()
        assert phase in avbl_phase, \
            f"The phase argument must be in {avbl_phase}."
        phase_df = self.history_df[self.history_df['phase'] == phase]
        return phase_df['ep'].unique().tolist()

    def get_episode_df(self, episode, phase='eval', iteration=None):
        episode_id = self.get_phase_eps(phase)[episode]
        filtered_df = self.history_df[
            np.logical_and(self.history_df['phase'] == phase,
                           self.history_df['ep'] == episode_id)]
        if iteration is not None:
            iter_idx = filtered_df['iteration'].unique().tolist()[iteration]
            filtered_df = filtered_df[filtered_df['iteration'] == iter_idx]
        return filtered_df

    def get_goal_distance(self):
        risk_distance = compute_risk_distance(*self.env_params['fire_dim'])
        return risk_distance + self.env_params['goal_threshold'] / 2.

    def get_ep_trajectories(self, episode, phase='eval', iteration=None):
        episode_df = self.get_episode_df(episode, phase)
        episode_iters = episode_df['iteration'].unique().tolist()

        if iteration is None:
            iteration_id = episode_iters
        elif type(iteration) is list:
            iteration_id = [episode_iters[i] for i in iteration]
        else:
            iteration_id = [episode_iters[iteration]]

        trajectories = list()
        for i in iteration_id:
            trj = dict()
            trj_df = episode_df[episode_df['iteration'] == i]
            s, a, s_t1 = self.get_tuple_control(trj_df)
            trj['success'] = (trj_df['final'] == 'goal_found').sum() >= 1
            trj['steps'] = len(trj_df)
            trj['length'] = np.linalg.norm(
                s_t1[:, :2] - s[:, :2], axis=1).sum()
            trj['rewards'] = trj_df['reward'].to_numpy()
            trj['timestamp'] = trj_df['timestamp'].to_numpy()
            trj['initial_pos'] = s[0, :3]
            trj['target_pos'] = trj_df[
                ['target_pos_x', 'target_pos_y', 'target_pos_z']
                ].head(1).to_numpy()[0]
            trj['states'] = s
            trj['actions'] = a
            trj['next_states'] = s_t1
            trj['target_dists'] = np.linalg.norm(
                trj['target_pos'] - trj['states'][:, :3], axis=1)
            trajectories.append(trj)
        return trajectories

    def iter_trajectory(self, trajectory):
        state, action, next_state = trajectory
        x_t = None
        u_t = None
        for step in range(len(state)):
            x_t = state[step] if step == 0 else state[step] - next_state[step]
            u_t = action[step].copy()
            yield x_t, u_t


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
        if sample[4] or sample[5]:
            self._iteration += 1
            self._stop_recording()


if __name__ == '__main__':
    experiment_path = Path('logs/test_2023-11-21_13-04-15/history.csv')
    exp_data = ExperimentData(experiment_path)
    trajectory_data = exp_data.get_ep_trajectories(0, [0])
    trajectory = trajectory_data[0]
    path_length, init_pos, target_pos, times, rewards, orientation = trajectory[:6]
    print('Trajectory with ', path_length, 'steps length, with ',
          rewards.sum(), 'of reward')
    print('\tfrom:', init_pos, 'to', target_pos)
    print('\ttimes:', times[0], 's - ', times[-1], 's',
          'total:', times[-1] - times[0], 'seconds')
    for trj in exp_data.iter_trajectory(trajectory[-3:]):
        pos, ang, speed, ang_vel = trj[0].reshape((4, 3))
        action = trj[1]
        print(pos, action)
