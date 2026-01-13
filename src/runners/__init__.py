REGISTRY = {}

from .rmappo_parallel_runner import ParallelRunner as RMAPPOParallelRunner
REGISTRY['rmappo_parallel'] = RMAPPOParallelRunner