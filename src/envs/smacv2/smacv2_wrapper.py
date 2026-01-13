from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from envs.smacv2.sc_cap_env_wrapper import StarCraftCapabilityEnvWrapper

from envs.multiagentenv import MultiAgentEnv

from absl import logging

class SMACv2Wrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

        self.env = StarCraftCapabilityEnvWrapper(**kwargs)
        self.episode_limit = self.env.episode_limit

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
        return rews, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_visibility_matrix(self):
        """Returns visibility matrix"""
        return self.env.get_visibility_matrix()
    
    def get_extrinsic_state(self):
        "new function for algorithm LAIES to get the extrinsic state"
        return self.env.get_extrinsic_state()
    
    def get_alive_agent(self):
        return self.env.get_alive_agent()

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""

        max_retries = 10
        retry_count = 0

        obss, _, is_success_init4unit = self.env.reset()
        while is_success_init4unit is False and retry_count < max_retries:
            self.env.close()
            self.env = StarCraftCapabilityEnvWrapper(**self.kwargs)
            obss, _, is_success_init4unit = self.env.reset()
            retry_count += 1
            logging.debug(f"Retrying reset after {retry_count} attempts")
        assert is_success_init4unit is True, f"Failed to reset environment after {max_retries} attempts"

        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()
    
    def get_ext_info(self):
        return self.env.get_ext_info()

    def get_stats(self):
        return self.env.get_stats()