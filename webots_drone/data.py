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
import time

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


def find_last_iteration(csv_file):
    import os

    with open(csv_file, 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return int(last_line.split(',')[2])


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

    def __init__(self, store_path, n_sensors=9, epsilon=False, extra_info=True,
                 other_cols=None):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_sensors = n_sensors
        self.epsilon = epsilon
        self._phase = 'init'
        self._ep = 0
        self._iteration = -1
        self.extra_info = extra_info
        self.other_cols = other_cols
        if self.store_path.is_file():
            print('WARNING:', self.store_path, 'already exists, adding data!')
            self._iteration = find_last_iteration(self.store_path)
        else:
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
        data_cols += ['penalization', 'bonus', 'final']

        for nid in range(self.n_sensors):
            data_cols += ['dist_sensor_' + str(nid)]

        data_cols += ['target_pos_x',
                      'target_pos_y',
                      'target_pos_z']
        data_cols += ['target_dim_height',
                      'target_dim_radius']

        if self.extra_info:
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
        if self.other_cols is not None:
            data_cols += self.other_cols

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
        if isinstance(sample[1], (list, tuple, np.ndarray)):
            action_str = ','.join(map(str, sample[1]))
            row.append(f'"[{action_str}]"')
        else:
            row.append(sample[1])
        row.append(sample[2])  # reward
        row.extend(state)  # next state
        row.append(sample[4])  # absorbing
        row.append(sample[5])  # last
        if self.epsilon is not False:
            row.append(self.epsilon())  # epsilon

        row.append(info['penalization'])
        row.append(info['bonus'])
        row.append(info['final'])

        for nid in range(self.n_sensors):
            row.append(info['dist_sensors'][nid])

        row.extend(info['target_position'])
        row.extend(info['target_dim'])

        if self.extra_info:
            row.extend(info['rc_position'])
            row.extend(info['emitter']['direction'])
            row.append(info['emitter']['signal_strength'])
            row.extend(info['motors_vel'])

        if self.other_cols is not None:
            for c in self.other_cols:
                row.append(info[c])

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
        self.set_quadrants()

    @property
    def quadrants(self):
        return self.env_params["target_quadrants"]

    @property
    def flight_area(self):
        return self.env_params["flight_area"]

    def join_eval_data(self, csv_regex=r'eval*/history_*.csv'):
        csv_paths = list(self.experiment_path.rglob(csv_regex))
        csv_paths.sort()
        eval_df = None
        if len(csv_paths) > 0:
            print(f"Loading {len(csv_paths)} evaluation data files")
        for csv_path in csv_paths:
            ep_df = pd.read_csv(csv_path)
            if eval_df is None:
                eval_df = ep_df.copy()
            else:
                eval_df = pd.concat((eval_df, ep_df))
        # print('eval_df', eval_df['phase'].describe())
        if eval_df is not None:
            # print('appending eval')
            self.history_df = pd.concat((self.history_df, eval_df))
            self.history_df = self.history_df.reset_index(drop=True)

    def get_reward_curve(self, phase='eval', by_quadrant=False):
        phase_df = self.history_df[self.history_df['phase'] == phase]
        if by_quadrant:
            if 'target_quadrant' not in self.history_df.columns:
                self.set_quadrants()
            reward_df = phase_df.groupby(['target_quadrant', 'ep']
                                         ).agg({'reward': ['sum', 'last']})

            # Flatten the MultiIndex columns
            reward_df.columns = ['sum', 'last']

            # Ensure unique values
            unique_quadrants = phase_df['target_quadrant'].unique()
            unique_eps = phase_df['ep'].unique()

            # Create index mappings
            quadrant_index = {q: i for i, q in enumerate(unique_quadrants)}
            ep_index = {e: i for i, e in enumerate(unique_eps)}

            # Initialize the NumPy array
            shape = (len(unique_quadrants), len(unique_eps), 2)
            reward_values = np.zeros(shape)

            # Populate the NumPy array
            for (quadrant, ep), row in reward_df.iterrows():
                q_idx = quadrant_index[quadrant]
                e_idx = ep_index[ep]
                reward_values[q_idx, e_idx, 0] = row['sum']
                reward_values[q_idx, e_idx, 1] = row['last']
        else:
            reward_values = phase_df.groupby('ep').agg(
                {'reward': ['sum', 'last']}).to_numpy()

        return reward_values

    def target_pos2quadrant(self, target_pos):
        target_pos = np.asarray(target_pos)
        return (self.quadrants == target_pos).sum(axis=1).argmax()

    def set_quadrants(self):
        target_quadrant = self.history_df.groupby(
            ['target_pos_x', 'target_pos_y', 'target_pos_z']).agg({'unique'})
        # target_quadrant = a_df.groupby(
        #     ['target_pos_x', 'target_pos_y', 'target_pos_z']).agg({'unique'})
        self.history_df['target_quadrant'] = -1
        for i, tpos in enumerate(target_quadrant.index.to_numpy()):
            df_query = (f"target_pos_x=={tpos[0]} &" +
                        f" target_pos_y=={tpos[1]} &" +
                        f" target_pos_z=={tpos[2]}")
            quadrant_df = self.history_df.query(df_query)
            self.history_df.loc[
                quadrant_df.index, "target_quadrant"] = self.target_pos2quadrant(tpos)

    def get_tuple_control(self, filtered_df):
        def orientation_correction(angle):
            """Apply UAV sensor offset."""
            angle += np.pi / 2.
            if angle > 2 * np.pi:
                angle -= 2 * np.pi
            return angle

        state_cols = ['pos_x', 'pos_y', 'pos_z',
                      'ori_x', 'ori_y', 'ori_z',
                      'vel_x', 'vel_y', 'vel_z',
                      'velang_x', 'velang_y', 'velang_z',
                      'north_rad']
        action_col = 'action'
        next_state_cols = ['next_' + sc for sc in state_cols]

        filtered_df.loc[:, 'north_rad'] = filtered_df['north_rad'].map(
            orientation_correction).copy()
        filtered_df.loc[:, 'next_north_rad'] = filtered_df['next_north_rad'].map(
            orientation_correction).copy()
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
            trj['target_quadrant'] = trj_df['target_quadrant'].head(1).to_numpy()[0]
            # trj['target_quadrant'] = self.target_pos2quadrant(
            #     trj['target_pos'])
            trj['states'] = s
            trj['actions'] = a
            trj['next_states'] = s_t1
            trj['target_dists'] = np.linalg.norm(
                trj['target_pos'] - trj['states'][:, :3], axis=1)
            trajectories.append(trj)
        return trajectories

    def get_episodes_info(self, phases=['learn', 'eval']):
        ep_info = dict()
        for ph in phases:
            phase_info = dict()
            try:
                episodes = self.get_phase_eps(ph)
                tries = [len(self.get_ep_trajectories(e, ph))
                         for e in episodes]
                rewards = self.get_reward_curve(ph)
            except AssertionError:
                episodes = list()
                tries = list()
                rewards = list()
            finally:
                phase_info['eps'] = episodes
                phase_info['tries'] = tries
                phase_info['rewards'] = rewards
                ep_info[ph] = phase_info
        return ep_info

    def get_info(self):
        state_data = list()
        if self.env_params['is_pixels']:
            state_data.append('RGB')
        if self.env_params['is_vector']:
            state_data.extend(self.env_params['uav_data'])
        for extra_key in ['target_pos2obs', 'target_dist2obs', 'target_dim2obs',
                          'action2obs']:
            if self.env_params[extra_key]:
                state_data.append(extra_key)
        if self.env_params['is_multimodal']:
            env_mode = 'multimodal'
        elif self.env_params['is_pixels']:
            env_mode = 'pixel-based'
        elif self.env_params['is_vector']:
            env_mode = 'vector-based'

        exp_info = dict(
            data_path=self.experiment_path,
            mode=env_mode,
            is_srl=self.agent_params['is_srl'],
            fix_target_pos=self.env_params['target_pos'] is not None,
            state_data=state_data,
            seed=self.train_params['seed']
            # epsilon=(self.agent_params["epsilon_start"],
            #          self.agent_params["epsilon_end"],
            #          self.agent_params["epsilon_steps"],)
            )

        phases_info = self.get_episodes_info()
        for k, v in phases_info.items():
            exp_info[k + '_eps'] = len(v['eps'])
            exp_info[k + '_tries'] = sum(v['tries'])
            exp_info[k + '_rewards'] = sum(v['rewards'])

        return exp_info

    def compute_ep_nav_metrics(self, ep, phase='eval'):
        ep_trajectories = self.get_ep_trajectories(ep, phase)
        trj_summ = np.zeros((len(ep_trajectories), 4))
        for i, trj_data in enumerate(ep_trajectories):
            trj_summ[i, 0] = trj_data['success']
            trj_summ[i, 1] = trj_data['length']
            short_dist = np.linalg.norm(
                trj_data['target_pos'] - trj_data['initial_pos'])
            trj_summ[i, 2] = short_dist - self.get_goal_distance()
            trj_summ[i, 3] = trj_data['target_dists'][-1] -\
                self.get_goal_distance()

        # Success Rate (SR)
        sr = trj_summ[:, 0].mean()
        # Success Path Length (SPL)
        max_short_path = trj_summ[:, 2] > trj_summ[:, 1]
        trj_summ[max_short_path, 1] = trj_summ[max_short_path, 2]
        spl = (trj_summ[:, 0] * (trj_summ[:, 2] / trj_summ[:, 1])).mean()
        # Soft SPL
        soft_spl = (trj_summ[:, 2] / trj_summ[:, 1]).mean()
        # Distance to Success (DTS)
        dts = trj_summ[:, 3].mean()
        return dict(tries=len(ep_trajectories), SR=sr, SPL=spl, SSPL=soft_spl,
                    DTS=dts)

    def get_nav_metrics(self, phase='eval'):
        episodes = self.get_phase_eps(phase)
        metrics = list()
        for ep in episodes:
            metrics.append(self.compute_ep_nav_metrics(ep, phase))
        return metrics

    def get_epsilon_curve(self):
        epsilon=(self.agent_params["epsilon_start"],
                 self.agent_params["epsilon_end"],
                 self.agent_params["epsilon_steps"],)
        diff = epsilon[1] - epsilon[0]
        diff /= self.agent_params["epsilon_steps"]
        eps_values = np.linspace(*epsilon)
        return eps_values

    def get_mem_beta_curve(self):
        mem_beta=(self.agent_params["memory_buffer"]["beta"],
                  1.,
                  self.agent_params["memory_buffer"]["beta_steps"],)
        diff = mem_beta[1] - mem_beta[0]
        diff /= self.agent_params["epsilon_steps"]
        beta_values = np.linspace(*mem_beta)
        return beta_values


class VideoCallback:
    def __init__(self, store_path, mdp, video_speed=1):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.env_sim = mdp.env.sim
        self._recording = False
        self._iteration = 0
        self.video_speed = video_speed

    @property
    def is_ready(self):
        return self.env_sim.movieIsReady()

    def start_recording(self, vid_name):
        vid_path = self.store_path / vid_name
        if vid_path.is_file():
            print('WARNING:', vid_path, 'already exists, overwriting!')

        while not self.is_ready:
            print('Previous encoder is still running....')
            time.sleep(.5)

        self.env_sim.movieStartRecording(str(vid_path.absolute()),
                                         width=848, height=480,
                                         quality=100, codec=0,
                                         acceleration=self.video_speed, caption=True)
        if self.env_sim.movieFailed():
            print('Error on recording movie...')
        else:
            self._recording = True

    def stop_recording(self):
        self.env_sim.movieStopRecording()
        self._recording = False

    def __call__(self, sample, info):
        if not self._recording:
            vid_path = self.store_path / f"iteration_{self._iteration:03d}.mp4"
            self.start_recording(vid_path)
        if sample[4] or sample[5]:
            self._iteration += 1
            self.stop_recording()


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
