#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:18:56 2024

@author: Angel Ayala
From OpenAI Baseline.
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
import gym
import numpy as np
from gym import spaces
from collections import deque


class ObservationStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        if len(shp) == 1:
            shp = (1, shp[0])
            obs_low = obs_low[np.newaxis, ...]
            obs_high = obs_high[np.newaxis, ...]
        obs_low = np.repeat(obs_low, k, axis=0)
        obs_high = np.repeat(obs_high, k, axis=0)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high,
            shape=((shp[0] * k,) + shp[1:]), dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if len(obs.shape) == 1:
            obs = obs[np.newaxis, ...]
        for _ in range(self.k):
            self.frames.append(obs)
        return self.observation(None), info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        if len(obs.shape) == 1:
            obs = obs[np.newaxis, ...]
        self.frames.append(obs)
        return self.observation(None), reward, done, trunc, info

    def observation(self, observation):        
        assert len(self.frames) == self.k
        return LazyArray(list(self.frames))


class LazyArray(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim]

    def frame(self, i):
        return self._force()[..., i]
