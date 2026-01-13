def test_alg_config_supports_reward(args):
    """
    Check whether algorithm supports specified reward configuration
    """
    if args.common_reward:
        # all algorithms support common reward
        return True
    else:
        if args.learner in ["coma_learner", "qtran_learner"]:
            # COMA and QTRAN only support common reward
            return False
        elif args.learner == "q_learner" and (
            args.mixer == "vdn" or args.mixer == "qmix"
        ):
            # VDN and QMIX only support common reward
            return False
        else:
            return True

def test_env_config_supports_reward(args):
    """
    Check whether environment supports specified reward configuration
    """
    if args.common_reward:
        # all environments support common reward
        return True
    else:
        if args.env in ["sc2_v1", "sc2_v2"]:
            # simple envs do not support common reward
            return False
        else:
            return True