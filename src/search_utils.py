import numpy as np

def get_hp_dist_cifar10(args):
    if args.continuous_search or args.continuous_search_drl:
        if args.continuous_search:
            from src.hp_opt_cs import learnable_highd_gaussian_continuous, get_hyper_loss
        elif args.continuous_search_drl:
            from src.hp_opt_cs_drl import learnable_highd_gaussian_continuous, get_hyper_loss
        # Define RL search space
        hyperparams_points = []
        if args.search_lr:
            hyperparams_points += [[0.0005, 0.05]]
        if args.search_ne:
            n_iterations_points = [[2, 5, 10, 15, 20, 25, 30, 35, 40]]
            hyperparams_points += n_iterations_points
        if args.search_aw:
            tmp_list = [np.linspace(0.1, 1.0, 10) for _ in range(len(args.dataset))]
            hyperparams_points += tmp_list
        if args.search_slr:
            hyperparams_points += [np.linspace(0.5, 1.5, 9)]
        np.random.seed(1)
        if args.continuous_search_drl:
            hparam_distro = learnable_highd_gaussian_continuous(hyperparams_points,
                                                                initial_precision=args.initial_precision,
                                                                fixed_precision=False, dirichlet=False,
                                                                n_clients=len(args.dataset), rl_nettype=args.rl_nettype)
        else:
            hparam_distro = learnable_highd_gaussian_continuous(hyperparams_points,
                                                                initial_precision=args.initial_precision,
                                                                fixed_precision=False, dirichlet=False,
                                                                n_clients=len(args.dataset))
    else:
        raise NotImplementedError

    return hparam_distro, get_hyper_loss