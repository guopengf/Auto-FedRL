import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math
import random
from helpers.utils import get_dataloader, get_train_dataset
import copy
import torch.nn.functional as F

### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNN(nn.Module):
    def __init__(self):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

class MyClient_cifar10:
    def __init__(self, args, dataset_id, client_id, net_dataidx_map,serial=False):
        self.args = args
        self.dataset_id = dataset_id

        self.client_id = client_id
        self.gpu_id = client_id%args.n_gpu

        if serial:
            self.gpu_id = 0
        self.weights = None

        self.init_model()

        self.init_loaders(client_id, net_dataidx_map)
        self.init_optimizer()

    def init_model(self):
        if self.args.net_type == 0:
            self.model = ModerateCNN().cuda(self.gpu_id)
        else:
            raise NotImplementedError

    def init_loaders(self, net_id, net_dataidx_map):
        args = self.args
        dataidxs = net_dataidx_map[net_id]
        self.ds_len = len(dataidxs)
        self.train_loader, self.val_loader = get_dataloader('cifar10', args.data_dir, args.batch_size, args.batch_size, dataidxs)
        print('client {}: train using {}, test using {} data points'.format(net_id,len(self.train_loader.dataset),len(self.val_loader.dataset)))

    
    def init_optimizer(self):
        if self.args.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = 0.0, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def train_round(self, cur_round,given_lr=None, given_epoch_round=None):
        self.model.train()
        args = self.args
        iters, epochs = 0, 0
        loss_function = nn.CrossEntropyLoss()
        writer = open(os.path.join(self.args.exp_dir, self.args.exp_name, 'client_%d.log' % self.client_id), 'a')
        writer.write("Starting round %d...\n" % cur_round)
        if args.refresh_optimizers:
            self.init_optimizer()
        self.adjust_lr(cur_round,given_lr)
        cur_n_epoch_round = args.n_epoch_round
        if given_epoch_round is not None:
            cur_n_epoch_round = given_epoch_round
        print('Train {} local epoch for client {}'.format(cur_n_epoch_round,self.client_id))
        print("lr: %.5f\n" % (self.optimizer.param_groups[-1]['lr']))

        main_model=copy.deepcopy(self.model)
        if args.epoch_mode:
            while epochs < cur_n_epoch_round:
                for batch_data in self.train_loader:
                    inputs, labels = batch_data[0].cuda(self.gpu_id), batch_data[1].cuda(self.gpu_id)

                    outputs = self.model(inputs)
                    loss = loss_function(outputs, labels)
                    weight_divergence_cost = 0.0
                    entropy_cost =0.0
                    if (args.weight_divergence_coeff > 0.0):
                        for own_p, main_p in zip(self.model.parameters(), main_model.parameters()):
                            weight_divergence_cost += args.weight_divergence_coeff * ((own_p - main_p) ** 2).sum()
                    if (args.entropy_coeff > 0.0):
                        probs_output = torch.exp(outputs) / (torch.exp(outputs).sum(1).view(-1, 1))
                        entropy = -(probs_output * torch.log(probs_output)).sum(1).mean()
                        entropy_cost = args.entropy_coeff * F.relu(args.entropy_threshold - entropy)

                    self.optimizer.zero_grad()
                    (loss + entropy_cost + weight_divergence_cost).backward()
                    self.optimizer.step()
                    if iters % 10 == 0:
                        writer.write("iteration %d, train_loss: %.4f lr: %.5f\n" % (iters + cur_round * args.n_epoch_round * self.ds_len / args.batch_size, loss.item(), self.optimizer.param_groups[-1]['lr']))
                    iters += 1
                epochs += 1
        writer.close()

    def validation_round(self, cur_round, client_dict):
        args = self.args
        best_metric = client_dict['best_metric']
        best_metric_round = client_dict['best_metric_round']
        self.model.eval()
        metric_values = list()
        model_path = os.path.join(self.args.exp_dir, args.exp_name, 'best_model%d.pth' % self.client_id)
        writer = open(os.path.join(self.args.exp_dir, args.exp_name, 'client_%d.log' % self.client_id), 'a')
        tbwriter = SummaryWriter(os.path.join(self.args.exp_dir, self.args.exp_name, 'runs_c%d/' % self.client_id))
        with torch.no_grad():

            correct, total = 0, 0
            for val_data in self.val_loader:
                val_images, val_labels = val_data[0].cuda(self.gpu_id), val_data[1].cuda(self.gpu_id)

                outputs = self.model(val_images)
                _, pred_label = torch.max(outputs.data, 1)

                total += val_images.data.size()[0]
                correct += (pred_label == val_labels.data).sum().item()

            metric = correct/float(total)
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_round = cur_round+1
                torch.save(self.model.state_dict(), model_path)
                writer.write("saved new best metric model\n")
            writer.write(
                "current round: {} current mean acc: {:.4f} best mean acc: {:.4f} at round {}\n".format(
                    cur_round + 1, metric, best_metric, best_metric_round
                )
            )
            tbwriter.add_scalar('val', metric, cur_round + 1)
        self.model.train()
        writer.close()
        tbwriter.close()
        client_dict['best_metric'] = best_metric
        client_dict['best_metric_round'] = best_metric_round

    
    def val_server(self):
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        for i, (input, target) in enumerate(self.val_loader):
            target = target.cuda(self.gpu_id)
            input = input.cuda(self.gpu_id)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg
    
    def load_weights(self, state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v.cuda(self.gpu_id)
        self.model.load_state_dict(state_dict)
    
    def adjust_lr(self, r, given_lr=None):
        if self.args.lr_schedule == 'constant':
            cur_lr = self.args.lr
        elif self.args.lr_schedule == 'cosine':
            lr_max = self.args.lr
            lr_min = 0
            cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(r / self.args.n_round * math.pi))
        elif self.args.lr_schedule == 'step':
            cur_lr = self.args.lr
            for s in self.args.lr_step:
                if r >= s:
                    cur_lr = cur_lr * 0.1
        if given_lr is not None: # mannual adjust by given lr
            cur_lr = given_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MyServer_cifar10:
    def __init__(self, args, dataset_id=0, client_id=0, net_dataidx_map=None, serial=True):
        self.args = args
        self.dataset_id = dataset_id

        self.client_id = client_id
        self.gpu_id = client_id

        if serial:
            self.gpu_id = 0
        self.weights = None

        self.init_model()
        self.init_loaders(client_id, net_dataidx_map)
        self.init_optimizer()


    def init_model(self):
        if self.args.net_type == 0:
            self.model = ModerateCNN().cuda(self.gpu_id)
        else:
            raise NotImplementedError

    def init_loaders(self, net_id, net_dataidx_map):
        args = self.args
        if net_dataidx_map is not None:
            dataidxs = []
            for k, v in net_dataidx_map.items():
                dataidxs += v
            self.ds_len = len(dataidxs)
            _, self.test_loader= get_dataloader('cifar10', args.data_dir, args.batch_size, args.batch_size, dataidxs)
            datasets_val = get_train_dataset(dataset='cifar10', datadir=args.data_dir,dataidxs=dataidxs)
            imgs = torch.stack([datasets_val[i][0] for i in range(len(datasets_val))])
            labels = torch.Tensor([datasets_val[i][1] for i in range(len(datasets_val))]).long()
            self.loss_eval_data = (imgs, labels)
            print('Server evaluating val loss using {} data points'.format(imgs.size(0)))

        else:
            dataidxs = None
            _, self.test_loader = get_dataloader('cifar10', args.data_dir, args.batch_size, args.batch_size,dataidxs)


    def init_optimizer(self):
        if self.args.optim_server == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr_server, momentum=self.args.momentum_server)
        elif self.args.optim_server == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.args.lr_server, eps=0.001)
        elif self.args.optim_server == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),betas=(self.args.momentum_server, self.args.sec_momentum_server), lr=self.args.lr_server, eps=0.001)
        elif self.args.optim_server == 'yogi':
            self.optimizer = optim.Yogi(self.model.parameters(), lr=self.args.lr_server, betas=(self.args.momentum_server, self.args.sec_momentum_server), eps=0.001, initial_accumulator=0.0)
        elif self.args.optim_server == 'novograd':
            self.optimizer = optim.NovoGrad(self.model.parameters(), lr=self.args.lr_server, betas=(self.args.momentum_server, self.args.sec_momentum_server), eps=0.001)

    def server_update(self, model_delta_dict, cur_round, given_lr=None):
        self.model.train()
        print("Starting server update round %d...\n" % cur_round)

        self.adjust_lr(cur_round, given_lr)
        self.optimizer.zero_grad()

        # Apply the update to the model. We must multiply weights_delta by -1.0 to
        # view it as a gradient that should be applied to the server_optimizer.
        for name, param in self.model.named_parameters():
            param.grad = -1.0*model_delta_dict[name].cuda(self.gpu_id)
        self.optimizer.step()

        print("Round %d, lr: %.5f\n" % (cur_round, self.optimizer.param_groups[-1]['lr']))

        return self.model.state_dict()


    def validation_round(self, cur_round, client_dict):
        args = self.args
        best_metric = client_dict['best_metric']
        best_metric_round = client_dict['best_metric_round']
        self.model.eval()
        metric_values = list()
        model_path = os.path.join(self.args.exp_dir, args.exp_name, 'best_model%d.pth' % self.client_id)
        writer = open(os.path.join(self.args.exp_dir, args.exp_name, 'client_%d.log' % self.client_id), 'a')
        tbwriter = SummaryWriter(os.path.join(self.args.exp_dir, self.args.exp_name, 'runs_c%d/' % self.client_id))
        with torch.no_grad():

            correct, total = 0, 0
            for val_data in self.val_loader:
                val_images, val_labels = val_data[0].cuda(self.gpu_id), val_data[1].cuda(self.gpu_id)

                outputs = self.model(val_images)
                _, pred_label = torch.max(outputs.data, 1)

                total += val_images.data.size()[0]
                correct += (pred_label == val_labels.data).sum().item()

            metric = correct / float(total)
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_round = cur_round + 1
                torch.save(self.model.state_dict(), model_path)
                writer.write("saved new best metric model\n")
            writer.write(
                "current round: {} current mean acc: {:.4f} best mean acc: {:.4f} at round {}\n".format(
                    cur_round + 1, metric, best_metric, best_metric_round
                )
            )
            tbwriter.add_scalar('val', metric, cur_round + 1)
        self.model.train()
        writer.close()
        tbwriter.close()
        client_dict['best_metric'] = best_metric
        client_dict['best_metric_round'] = best_metric_round

    def get_rl_val_server(self,v_losses,v_accuracies):
        v_losses.append(v_losses[-1].copy())
        v_accuracies.append(v_accuracies[-1].copy())
        args = self.args
        self.model.eval()
        criterion = nn.CrossEntropyLoss().cuda(self.gpu_id)
        correct = 0.0
        total = 0
        batch_idx = 0
        total_loss = 0.0

        with torch.no_grad():
            val_images = self.loss_eval_data[0].cuda(self.gpu_id)
            val_labels = self.loss_eval_data[1].cuda(self.gpu_id)
            outputs = self.model(val_images)
            loss = criterion(outputs, val_labels)
            total_loss += loss.item()
            _, pred_label = torch.max(outputs.data, 1)
            batch_idx += 1
            total += val_images.data.size()[0]
            correct += (pred_label == val_labels.data).sum().item()
            metric = correct / float(total)
        val_loss = total_loss/batch_idx
        for model_idx in range(self.args.num_clients):
            v_losses[-1][model_idx] = val_loss
            v_accuracies[-1][model_idx] = metric

    def val_server(self):
        args = self.args
        self.model.eval()
        with torch.no_grad():
            correct = 0.0
            total = 0
            for val_data in self.val_loader:
                val_images, val_labels = val_data[0].cuda(self.gpu_id), val_data[1].cuda(self.gpu_id)
                outputs = self.model(val_images)
                _, pred_label = torch.max(outputs.data, 1)

                total += val_images.data.size()[0]
                correct += (pred_label == val_labels.data).sum().item()
            metric = correct / float(total)
        return metric

    def load_weights(self, state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v.cuda(self.gpu_id)
        self.model.load_state_dict(state_dict)

    def adjust_lr(self, r, given_lr=None):
        if self.args.lr_schedule_server == 'constant':
            cur_lr = self.args.lr_server
        elif self.args.lr_schedule_server == 'cosine':
            lr_max = self.args.lr_server
            lr_min = 0
            cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(r / self.args.n_round * math.pi))
        elif self.args.lr_schedule_server == 'step':
            cur_lr = self.args.lr_server
            for s in self.args.lr_step_server:
                if r >= s:
                    cur_lr = cur_lr * 0.1
        if given_lr is not None: # mannual adjust by given lr
            cur_lr = given_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr


def init_all_clients(clients, global_model=None):
    with torch.no_grad():
        if global_model is None:
            # if global_model is not provided, just use one of clients models to initialize all others
            global_model = copy.deepcopy(clients[0].model.state_dict())

        avg_dict = {}
        for k, v in global_model.items():
            avg_dict[k] = v.cpu()

    for client in clients:
        client.load_weights(avg_dict)

    return global_model

def weighted_model_avg(clients, weight, update_client=True):
    with torch.no_grad():
        print('averaging models... weight: %s' % str(weight))
        avg_dict = {}
        for k, v in clients[0].model.state_dict().items():
            avg_dict[k] = torch.zeros(v.size(), dtype=v.dtype, layout=v.layout)

        for cid in range(len(clients)):
            for k, v in clients[cid].model.state_dict().items():
                avg_dict[k] += v.cpu() * weight[cid]

    if update_client:
        for client in clients:
            client.load_weights(avg_dict)
    
    return avg_dict

def weighted_model_avg_with_server(clients,server, weight, update_client=True):
    print('averaging models... weight: %s' % str(weight))

    trained_params = [list(m.model.parameters()) for m in clients]
    transposed_trained_params = [*zip(*trained_params)]
    for param_set, main_p in zip(transposed_trained_params, server.model.parameters()):
        main_p.data.add_(
            sum([(m.cuda(server.gpu_id)-main_p) * weight for m, weight in zip(param_set, weight)]))
    avg_dict = copy.deepcopy(server.model.state_dict())
    if update_client:
        for client in clients:
            client.load_weights(avg_dict)
    return avg_dict


def weighted_model_avg_with_server_update(server, global_model, clients, weight, r, given_lr=None,update_client=True):
    with torch.no_grad():
        print('update models... weight: %s' % str(weight))
        model_delta_dict = {}

        # init delta dict
        for k, v in clients[0].model.state_dict().items():
            model_delta_dict[k] = torch.zeros(v.size(), dtype=v.dtype, layout=v.layout)

        # collect weighted model delta from each clients
        for cid in range(len(clients)):
            for k, v in clients[cid].model.state_dict().items():
                model_delta_dict[k] += weight[cid] * (v.cpu() - global_model[k].cpu())

    updated_server_dict = server.server_update(model_delta_dict, r, given_lr)
    with torch.no_grad():
        avg_dict = {}
        for k, v in updated_server_dict.items():
            avg_dict[k] = v.cpu()

    if update_client:
        for client in clients:
            client.load_weights(avg_dict)

        return avg_dict

def model_avg(clients):
    print('averaging models... FedAvg')
    avg_dict = {}
    for k, v in clients[0].model.state_dict().items():
        avg_dict[k] = torch.zeros(v.size(), dtype=v.dtype, layout=v.layout)
    for cid in range(len(clients)):
        for k, v in clients[cid].model.state_dict().items():
            avg_dict[k] += v.cpu() / len(clients)
    
    for client in clients:
        client.load_weights(avg_dict)
    
    return avg_dict
