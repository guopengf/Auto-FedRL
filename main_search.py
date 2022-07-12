import torch.multiprocessing as mp
from helpers.utils import partition_data, print_options, save_checkpoint
from src.server_utils import val_server
import time
from src.single_net_cifar10_search import *
import random
from src.search_utils import get_hp_dist_cifar10
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='train a vgg-9 on cifar 10')
parser.add_argument('--data_dir', type=str, default='../../dataset')
parser.add_argument('--exp_name', type=str, default='debug')
parser.add_argument('--exp_dir', type=str, default='../../exps/exp')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--dataset', type=str, default="1,2,3")
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_schedule', type=str, default='constant')
parser.add_argument('--lr_step', type=list, default=[30,40])
parser.add_argument('--n_round', type=int, default=200)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--server_interval', type=int, default=1)
parser.add_argument('--epoch_mode', action='store_true')
parser.add_argument('--n_epoch_round', type=int, default=20)
parser.add_argument('--weighting_mode', type=int, default=1)
parser.add_argument('--init_exp', type=str, default='')
parser.add_argument('--net_type', type=int, default=0)
parser.add_argument('--even_init', action='store_true')
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--init_with_same_weight', action='store_true')
# For FedOpt
parser.add_argument('--FedOpt', action='store_true')
parser.add_argument('--optim_server', type=str, default='sgd')
parser.add_argument('--momentum_server', type=float, default=0.0)
parser.add_argument('--sec_momentum_server', type=float, default=0.0)
parser.add_argument('--lr_server', type=float, default=1.0)
parser.add_argument('--lr_schedule_server', type=str, default='constant')
parser.add_argument('--lr_step_server', type=list, default=[30,40])
parser.add_argument('--seed', type=int, default=0)
# For REINFORCE Search
parser.add_argument('--Search', action='store_true')
parser.add_argument('--hyper_lr', type=float, default=0.01, help='Hyper learning rate')
parser.add_argument('--server_gpu', default=0, type=int, help='which gpu to host server')
parser.add_argument('--search_aw', action='store_true')
parser.add_argument('--search_slr', action='store_true')
parser.add_argument('--search_lr', action='store_true')
parser.add_argument('--search_ne', action='store_true')
parser.add_argument('--n-val-iters-per-round', default=10, type=int,help='number of validation mini-batches per client in each round')
parser.add_argument('--initial_precision', type=float, default=85.0, help='precision of gaussian')
parser.add_argument('--baseline_cutoff_interval', default=5, type=int,
                    help='number of rounds to consider for the REINFORCE baseline (Z in the paper)  ')
parser.add_argument('--no-reinforce-baseline',  action='store_true',
                    help='Do not use a baseline in the online reinforce algorithm (default: False)')
parser.add_argument('--windowed_updates',  action='store_true',
                    help='Update using all rewards/actions in the cutoff window  (default: False)')
parser.add_argument('--entropy-coeff', default=1.0, type=float, help='Coefficient of the entropy regularization term')
parser.add_argument('--entropy-threshold', default=2.0, type=float, help='Entropy threshold H_{min} used in the entropy regularization term')
parser.add_argument('--weight-divergence-coeff', default=0.0, type=float, help='FedProx: coefficient of the weight divergence loss')
parser.add_argument('--debug',  action='store_true',help='debug mode')
parser.add_argument('--refresh-optimizers',  action='store_true',
                    help='Reinitialize optimizers after every round   (default: False)')
parser.add_argument('--continuous_search',  action='store_true',help='continuous search space')
parser.add_argument('--continuous_search_drl',  action='store_true',help='continuous search space using deep RL')
parser.add_argument('--rl_nettype',  type=str, default='mlp',help='the network typr for deep RL')

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger('nibabel.global').level = 36
    mp.set_start_method('spawn')

    args = parser.parse_args()
    args.dataset = [int(x) for x in args.dataset.split(",")]
    args.num_clients = len(args.dataset)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    exp_name = args.exp_name
    os.makedirs(os.path.join(args.exp_dir, exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, exp_name, 'cmodels/'), exist_ok=True) # server model save path
    print_options(parser, args)

    y_train, net_dataidx_map, traindata_cls_counts = partition_data(dataset='cifar10', datadir=args.data_dir, logdir= os.path.join(args.exp_dir, exp_name),
                       partition='hetero-dir', n_nets=len(args.dataset), alpha=0.5,)
        
    clients = []
    manager = mp.Manager()
    client_dicts = []
    global_model = None
    server_dict = manager.dict()
    server_dict['best_metric'] = -1
    server_dict['best_metric_round'] = -1

    if args.n_gpu > 1:
        serial = False
    else:
        # serial = True lets train on a single GPU
        serial = True

    # we need split a small part of cifar-10 train data as val
    tmp_dict = net_dataidx_map
    net_dataidx_map = {}
    net_dataidx_map_val = {}
    for k, v in tmp_dict.items():
        if not args.debug:
            net_dataidx_map[k] = v[:]
            net_dataidx_map_val[k] = v[-args.n_val_iters_per_round * args.batch_size:]
        else:
            net_dataidx_map[k] = v[:500]
            net_dataidx_map_val[k] = v[-100:]
            
    if args.Search:
        hp_dist, get_hyper_loss = get_hp_dist_cifar10(args)
        hyper_optimizer = torch.optim.Adam(hp_dist.parameters(), args.hyper_lr, betas=(0.7, 0.7))
        for name, param in hp_dist.named_parameters():
            print(name)

    client_id = 0

    for cid in args.dataset:
        clients.append(MyClient_cifar10(args, cid, client_id, net_dataidx_map, serial=serial))
        md = manager.dict()
        md['best_metric'] = -1
        md['best_metric_round'] = -1
        client_dicts.append(md)
        client_id += 1

    server = MyServer_cifar10(args, client_id=args.server_gpu,net_dataidx_map=net_dataidx_map_val,serial=serial)


    if args.init_with_same_weight:
        print('init all nets with same weight')
        global_model = init_all_clients(clients, copy.deepcopy(server.model.state_dict()))


    if args.even_init:
        weight = [1.0 for client in clients]
    else:
        weight = [client.ds_len for client in clients]
    weight = [e / sum(weight) for e in weight]

    if len(args.init_exp) > 0:
        print('Init weight from %s' % args.init_exp)
        for client in clients:
            client.model.load_state_dict(torch.load(os.path.join('../../exps/', args.init_exp, 'best_model%d.pth' % client.dataset_id)), strict=False)

    logprob_history = []
    hparam_per_round_history = []
    sample_history = []
    val_losses = [np.ones(len(args.dataset)) * -np.inf]
    val_accuracies = [np.ones(len(args.dataset)) * -np.inf]
    hparams_distro_params_history = []
    weight_history = []
    
    lrate, train_iters_per_round, slr = None, None, None
    for r in range(args.n_round):
        if args.Search:
            if args.continuous_search or args.continuous_search_drl:
                hparam, logprob = hp_dist.forward()
            else:
                hparam, logprob, hparam_idx = hp_dist.forward()

            hparam_list = [i for i in hparam]
            if args.search_lr:
                lrate = hparam_list.pop(0)
            if args.search_ne:
                train_iters_per_round = hparam_list.pop(0)
                train_iters_per_round = int(train_iters_per_round + 0.5)
            if args.search_aw:
                aw = [hparam_list.pop(0) for _ in range(args.num_clients)]
                aw_tensor = torch.tensor([aw])
                aw_tensor = F.softmax(aw_tensor,dim=1)
                weight = [aw_tensor[:, i].item() for i in range(args.num_clients)]
            if args.search_slr:
                slr = hparam_list.pop(0)
            # add some constrains to prevent negative value
            if args.continuous_search or args.continuous_search_drl:
                if args.search_lr:
                    lrate = lrate if lrate > 0.0001 else 0.0001
                if args.search_ne:
                    train_iters_per_round = int(train_iters_per_round + 0.5) if train_iters_per_round >= 1 else 1
                if args.search_slr:
                    slr = slr if slr > 0.1 else 0.1

            logprob_history.append(logprob)
            weight_history.append(weight)
            sample_history.append(hparam)

            # Get val loss
            server.get_rl_val_server(val_losses,val_accuracies)

        print('Start training round %d...' % r)
        start_time = time.time()

        # train round
        processes = []
        for i, client in enumerate(clients):
            p = mp.Process(target=client.train_round, args=(r,lrate,train_iters_per_round))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        # val round
        if (r+1) % args.val_interval == 0:
            print('Start val round %d...' % r)
            processes = []
            for client, client_dict in zip(clients,client_dicts):
                p = mp.Process(target=client.validation_round, args=(r, client_dict))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        if args.Search:
            hyper_loss, imp = get_hyper_loss(val_losses, max(0, r - args.baseline_cutoff_interval),
                                             logprob_history, args.no_reinforce_baseline, args.windowed_updates)
            if not (type(hyper_loss) == int):
                hyper_optimizer.zero_grad()
                (-hyper_loss).backward(retain_graph=True)
                hyper_optimizer.step()
                # We need release the memory
                if args.windowed_updates and len(logprob_history) > args.baseline_cutoff_interval:
                    print("Release Memory......"*5)
                    release_list = logprob_history[:-args.baseline_cutoff_interval]
                    keep_list = logprob_history[-args.baseline_cutoff_interval:]
                    tmp_list = [logpro.detach() for logpro in release_list]
                    logprob_history = tmp_list + keep_list

                if args.continuous_search_drl:
                    hparams_distro_params_history.append([hp_dist.mean.data.cpu().numpy().copy(),hp_dist.precision_component.data.cpu().numpy().copy()])
                else:
                    hparams_distro_params_history.append(
                        [x.data.cpu().numpy().copy() for x in hp_dist.parameters()])

        # model aggregation
        if args.FedOpt:
            # We have a server update
            if args.weighting_mode == 1:
                global_model = weighted_model_avg_with_server_update(server, global_model, clients, weight, r, slr)
        elif args.Search:
            print('Search and weighted avg..')
            if args.weighting_mode == 1:
                global_model = weighted_model_avg_with_server(clients, server, weight)
        else:
            if args.weighting_mode == 0:
                global_model = model_avg(clients)
            elif args.weighting_mode == 1:
                global_model = weighted_model_avg(clients,  weight)

        # val center model
        if (r+1) % args.server_interval == 0 and args.weighting_mode >=0 and args.weighting_mode < 5:
            p = mp.Process(target=val_server, args=(args, clients, global_model, server_dict, r))
            p.start()
            p.join()

        save_checkpoint({
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'hparams_distro_params_history': hparams_distro_params_history,
            'sample_history': sample_history,
            'logprob_history':[logpro.detach() for logpro in logprob_history],
            'weight_history':weight_history,
            'epoch': r + 1,
            'args': args
        }, save_dir=os.path.join(args.exp_dir, args.exp_name))

        end_time = time.time()
        print('time elapsed: %.4f' % (end_time - start_time))