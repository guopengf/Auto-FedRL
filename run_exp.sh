#!/bin/bash
#
# continuous search
python main_search.py --exp_name cs_c8_final --data_dir ../data --exp_dir ../results_release --dataset 1,2,3,4,5,6,7,8 --net_type 0 --val_interval 20 --server_interval 1 --n_gpu 2 --n_round 200 --n_epoch_round 20 --epoch_mode --weighting_mode 1 --init_with_same_weight --Search --FedOpt --continuous_search --search_lr --search_ne --search_aw --search_slr --hyper_lr 0.01 --initial_precision 85.0 --windowed_updates --server_gpu 1
# continuous search using mlp
#python main_search.py  --exp_name cs_mlp_c8_final --data_dir ../data --exp_dir ../results_release --dataset 1,2,3,4,5,6,7,8 --net_type 0 --val_interval 20 --server_interval 1 --n_gpu 2 --n_round 200 --n_epoch_round 20 --epoch_mode --weighting_mode 1 --init_with_same_weight --Search --FedOpt --continuous_search_drl --rl_nettype mlp --search_lr --search_ne --search_aw --search_slr --hyper_lr 0.01 --initial_precision 85.0 --windowed_updates --server_gpu 0
