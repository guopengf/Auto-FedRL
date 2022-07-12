import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def val_server(args, clients, model, server_dict, r):
    metric = []
    for client in clients:
        metric.append(client.val_server())
        break
    metric = np.mean(metric)
    tbwriter = SummaryWriter(os.path.join(args.exp_dir, args.exp_name, 'runs_server/'))
    tbwriter.add_scalar('val', metric, r + 1)
    writer = open(os.path.join(args.exp_dir, args.exp_name, 'server.log'), 'a')
    if metric > server_dict['best_metric']:
        torch.save(model, os.path.join(args.exp_dir, args.exp_name, 'cmodels', 'best_global_model.pth'))
        server_dict['best_metric'] = metric
        server_dict['best_metric_round'] = r + 1
    print("current round: {} current mean acc: {:.4f} best mean acc: {:.4f} at round {}\n".format(
                r + 1, metric, server_dict['best_metric'], server_dict['best_metric_round']
            ))
    writer.write(
        "current round: {} current mean acc: {:.4f} best mean acc: {:.4f} at round {}\n".format(
            r + 1, metric, server_dict['best_metric'], server_dict['best_metric_round']
        ))
    writer.close()
    tbwriter.close()