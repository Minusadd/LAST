import numpy as np
import torch
from .replay_buffer import ReplayBuffer
from project2 import pytorch_util as ptu

class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim, skill_dim
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, *observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        #self._next_obs = torch.zeros((max_replay_buffer_size, *observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._skills = np.zeros((max_replay_buffer_size, skill_dim))
        self._ms = np.zeros((max_replay_buffer_size, 2))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.clear()

    def add_sample(self, observation, action, skill, selected_m, reward, terminal,
                   **kwargs):

        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._skills[self._top] = skill
        self._ms[self._top] = selected_m
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        #self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._cur_episode_start = self._top
        self._episode_starts.append(self._cur_episode_start)


    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = [0]
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            skills=self._skills[indices],
            ms = self._ms[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            #next_observations=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_path(self, sequence=True):

        start = np.random.choice(self._episode_starts[:-1])
        pos_idx = self._episode_starts.index(start)
        indices = list(range(start, self._episode_starts[pos_idx + 1]))

        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            if i == 0:
                first_indices = list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices), self.sample_data(first_indices)

    def random_paths(self, batch_size):
        idx = 0
        obs = []
        acts = []
        skills = []
        dones = []
        rews = []
        ms = []
        next_obs = []
        while idx < batch_size:
            data = self.random_path()
            obs.append(ptu.from_numpy(data['observations']))
            acts.append(ptu.from_numpy(data['actions']))
            skills.append(ptu.from_numpy(data['skills']))
            ms.append(ptu.from_numpy(data['ms']))
            dones.append(ptu.from_numpy(data['terminals']))
            rews.append(ptu.from_numpy(data['rewards']))
            idx+=1
            # next_obs.append(data['next_observations'])
        return obs, acts, skills, ms, rews, dones

    def num_steps_can_sample(self):
        return self._size



class SimpleReplayBuffer2(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim, skill_dim
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, *observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        #self._next_obs = torch.zeros((max_replay_buffer_size, *observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._skills = np.zeros((max_replay_buffer_size, skill_dim))
        self._ms = np.zeros((max_replay_buffer_size, 2))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._masks = np.zeros((max_replay_buffer_size, 89))
        self.clear()

    def add_sample(self, observation, action, skill, selected_m, reward, terminal, mask,
                   **kwargs):

        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._skills[self._top] = skill
        self._ms[self._top] = selected_m
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._masks[self._top] = mask
        #self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._cur_episode_start = self._top
        self._episode_starts.append(self._cur_episode_start)


    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = [0]
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            skills=self._skills[indices],
            ms = self._ms[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            masks=self._masks[indices],
            #next_observations=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_path(self, sequence=True):

        start = np.random.choice(self._episode_starts[:-1])
        pos_idx = self._episode_starts.index(start)
        indices = list(range(start, self._episode_starts[pos_idx + 1]))

        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            if i == 0:
                first_indices = list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices), self.sample_data(first_indices)

    def random_paths(self, batch_size):
        idx = 0
        obs = []
        acts = []
        skills = []
        dones = []
        rews = []
        ms = []
        masks = []
        next_obs = []
        while idx < batch_size:
            data = self.random_path()
            obs.append(ptu.from_numpy(data['observations']))
            acts.append(ptu.from_numpy(data['actions']))
            skills.append(ptu.from_numpy(data['skills']))
            ms.append(ptu.from_numpy(data['ms']))
            dones.append(ptu.from_numpy(data['terminals']))
            rews.append(ptu.from_numpy(data['rewards']))
            masks.append(ptu.from_numpy(data['masks']))
            idx+=1
            # next_obs.append(data['next_observations'])
        return obs, acts, skills, ms, rews, dones, masks

    def num_steps_can_sample(self):
        return self._size


class SimpleReplayBuffer3(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim, skill_dim
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, *observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        #self._next_obs = torch.zeros((max_replay_buffer_size, *observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._skills = np.zeros((max_replay_buffer_size, skill_dim))
        self._ms = np.zeros((max_replay_buffer_size, 2))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._masks = np.zeros((max_replay_buffer_size, 89))
        self.clear()

    def add_sample(self, observation, action, reward, terminal, mask,
                   **kwargs):

        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._masks[self._top] = mask
        #self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._cur_episode_start = self._top
        self._episode_starts.append(self._cur_episode_start)


    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = [0]
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            masks=self._masks[indices],
            #next_observations=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_path(self, sequence=True):

        start = np.random.choice(self._episode_starts[:-1])
        pos_idx = self._episode_starts.index(start)
        indices = list(range(start, self._episode_starts[pos_idx + 1]))

        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            if i == 0:
                first_indices = list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices), self.sample_data(first_indices)

    def random_paths(self, batch_size):
        idx = 0
        obs = []
        acts = []
        skills = []
        dones = []
        rews = []
        ms = []
        masks = []
        next_obs = []
        while idx < batch_size:
            data = self.random_path()
            obs.append(ptu.from_numpy(data['observations']))
            acts.append(ptu.from_numpy(data['actions']))
            dones.append(ptu.from_numpy(data['terminals']))
            rews.append(ptu.from_numpy(data['rewards']))
            masks.append(ptu.from_numpy(data['masks']))
            idx+=1
            # next_obs.append(data['next_observations'])
        return obs, acts, rews, dones, masks

    def num_steps_can_sample(self):
        return self._size


class SimpleReplayBufferrs(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim, skill_dim
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, *observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        #self._next_obs = torch.zeros((max_replay_buffer_size, *observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._skills = np.zeros((max_replay_buffer_size, skill_dim))
        self._ms = np.zeros((max_replay_buffer_size, 2))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._srews = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.clear()

    def add_sample(self, observation, action, skill, selected_m, reward, terminal, s_rew,
                   **kwargs):

        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._skills[self._top] = skill
        self._ms[self._top] = selected_m
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._srews[self._top] = s_rew
        #self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._cur_episode_start = self._top
        self._episode_starts.append(self._cur_episode_start)


    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = [0]
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            skills=self._skills[indices],
            ms = self._ms[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            srews=self._srews[indices],
            #next_observations=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_path(self, sequence=True):

        start = np.random.choice(self._episode_starts[:-1])
        pos_idx = self._episode_starts.index(start)
        indices = list(range(start, self._episode_starts[pos_idx + 1]))

        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            if i == 0:
                first_indices = list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices), self.sample_data(first_indices)

    def random_paths(self, batch_size):
        idx = 0
        obs = []
        acts = []
        skills = []
        dones = []
        rews = []
        ms = []
        next_obs = []
        srews = []
        while idx < batch_size:
            data = self.random_path()
            obs.append(ptu.from_numpy(data['observations']))
            acts.append(ptu.from_numpy(data['actions']))
            skills.append(ptu.from_numpy(data['skills']))
            ms.append(ptu.from_numpy(data['ms']))
            dones.append(ptu.from_numpy(data['terminals']))
            rews.append(ptu.from_numpy(data['rewards']))
            srews.append(ptu.from_numpy(data['srews']))
            idx+=1
            # next_obs.append(data['next_observations'])
        return obs, acts, skills, ms, rews, dones, srews

    def num_steps_can_sample(self):
        return self._size