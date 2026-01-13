import numpy as np
import os
from collections.abc import Mapping
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import REGISTRY as run_REGISTRY

import warnings

warnings.simplefilter("ignore")
SETTINGS['CAPTURE_MODE'] = "fd" if os.name == 'nt' else "sys" # set to "no" if you want to see stdout/stderr in console

logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# os.environ['OMP_NUM_THREADS'] = '6'
# os.environ['MKL_NUM_THREADS'] = '6'
# th.set_num_threads(4)

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)

    if config.get('manual_seed', False):
        config['seed'] = config['manual_seed']

    np.random.seed(config["manual_seed"])
    th.manual_seed(config["manual_seed"])
    config['env_args']['seed'] = config["manual_seed"]
    
    # run
    if "use_per" in _config and _config["use_per"]:
        run_REGISTRY['per_run'](_run, config, _log)
    else:
        run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():               
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result

def set_other_config(params, key, config_dict, _type):
    # set config_dict with command line args
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == key:
            config_dict[key[2:]] = _type(_v.split("=")[1])
            del params[_i]
            break
    return config_dict

def main(params=None):
    if params is None:
        params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--alg-config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # set config_dict with command line args
    config_dict = set_other_config(params, "--manual_seed", config_dict, int)
    config_dict = set_other_config(params, "--cuda_id", config_dict, str)

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name']) + "_seed_{}".format(config_dict['manual_seed'])

    # SC2 does not need to set the local_results_path in sc2.yaml, while mpe needs to set the local_results_path in mpe.yaml
    if config_dict['env'] == 'sc2_v2':
        local_map_name = config_dict['local_name']
    else:
        local_map_name = map_name
    if 'local_results_path' not in config_dict:
        config_dict['local_results_path'] = local_map_name
    
    # Process the case of reinfroce_learner
    if 'learner' in config_dict and config_dict['learner'] == 'reinforce_learner':
        config_dict['batch_size'] = 1
        config_dict['buffer_size'] = 1
    
    # Process the case of episode
    if 'runner' in config_dict and config_dict['runner'] == 'episode':
        config_dict['batch_size_run'] = 1
    
    if 'evaluate' in config_dict and config_dict['evaluate']:
        config_dict['runner'] = 'test_episode'
        config_dict['batch_size_run'] = 1

    # Process the cuda device
    if 'use_cuda' in config_dict and config_dict['use_cuda']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict['cuda_id'])

    results_path = join(dirname(dirname(abspath(__file__))), "results/sacred", config_dict["local_results_path"])
    file_obs_path = join(results_path, local_map_name, algo_name)
    
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))

    # now add all the config to sacred
    ex.add_config(config_dict)
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

    # flush
    sys.stdout.flush()

if __name__ == '__main__':
    main()
