from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smacv2.env.starcraft2.maps import smac_maps

map_param_registry = {}

smac_maps.map_param_registry.update(map_param_registry)

def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]


for name in map_param_registry.keys():
    globals()[name] = type(name, (smac_maps.SMACMap,), dict(filename=name))
