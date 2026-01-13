from smacv2.env.starcraft2.distributions import get_distribution
from smacv2.env.starcraft2.starcraft2 import CannotResetException
from envs.smacv2.StarCraft2Env import StarCraft2Env
from smacv2.env import MultiAgentEnv

import numpy as np


class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = StarCraft2Env(**kwargs)
        assert (
            self.distribution_config.keys()
            == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"

    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key == "n_units" or env_key == "n_enemies":
                continue
            config["env_key"] = env_key
            # add n_units key
            config["n_units"] = self.distribution_config["n_units"]
            config["n_enemies"] = self.distribution_config["n_enemies"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def reset(self):
        try:
            reset_config = {}
            for distribution in self.env_key_to_distribution_map.values():
                reset_config = {**reset_config, **distribution.generate()}

            return self.env.reset(reset_config)
        except CannotResetException as cre:
            # just retry
            self.reset()

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()
    
    def get_visibility_matrix(self):
        return self.env.get_visibility_matrix()
    
    def get_extrinsic_state(self):
        "new function for algorithm to get the extrinsic state"
        return self.env.get_extrinsic_state()
    
    def get_alive_agent(self):
        return self.env.get_alive_agent()

    def get_env_info(self):
        return self.env.get_env_info()
    
    def get_ext_info(self):
        return self.env.get_ext_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()
