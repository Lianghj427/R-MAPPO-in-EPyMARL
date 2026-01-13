from functools import partial
from .multiagentenv import MultiAgentEnv
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)
REGISTRY = {}
def register_smacv2():
    from envs.smacv2 import smacv2
    REGISTRY["sc2_v2"] = partial(env_fn, env=smacv2)
    if sys.platform == 'linux':
        os.environ.setdefault("SC2PATH", "~/StarCraftII")