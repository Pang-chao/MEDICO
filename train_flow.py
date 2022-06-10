import os
import sys
# for linux env.
sys.path.insert(0,'..')
import torch
import torch.nn as nn
from model import MoFlow, rescale_adj
import time
from utils.timereport import TimeReport
import functools
from config import get_config, Hyperparameters
from torch.utils.tensorboard import SummaryWriter
from data.data_process import load_qm9, load_drugs, pick_data
from generate import load_model

print = functools.partial(print, flush=True)
config = get_config()
def train():
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    # Device configuration
    if config.gpu >= 0:
        device = torch.device('cuda:'+str(config.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        # cpu
        device = torch.device('cpu')

    if config.data_name == 'qm9':
        train_dataset = load_qm9(config.train_size, config.seed)
        atom_type = 5
        atom_max_num = 9
        bond_type = 4
    elif config.data_name == 'drugs':
        train_dataset = load_drugs(config.train_size, config.seed)
        atom_type = 8
        atom_max_num = 20
        bond_type = 4

    batch_size = config.batch_size
    learning_rate = config.learning_rate
    max_epochs = config.max_epochs
    if config.load_params == True:
        params_path = 'flow_params.json'
        model_params = Hyperparameters(path=params_path)
        snapshop_path = 'model_snapshot_epoch_340'
        model = load_model(snapshop_path, model_params, config.data_name)
    else:
        model_params = Hyperparameters()
        model = MoFlow(model_params, config.data_name)

    if torch.cuda.device_count() > 1 and config.multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model)
    model = model.to(device)

    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(time.time() - start))
    if config.gpu >= 0:
        print('Using GPU device:{}!'.format(config.gpu))
    print('Num batch-size: {}'.format(config.batch_size))
    print('Num epoch: {}'.format(max_epochs))
    print('==========================================')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()
    iter_per_epoch = len(train_dataset) / batch_size * 0.68
    print(iter_per_epoch)
    tr = TimeReport(total_iter=max_epochs * iter_per_epoch)
    for epoch in range(max_epochs):
        print("In epoch {}, Time: {}".format(epoch+1, time.ctime()))
        train_loader = pick_data(train_dataset, batch_size)
        for batch_data in train_loader:
            if batch_data.atom.size()[0] != batch_size * atom_max_num:
                break
            optimizer.zero_grad()
            x = batch_data.atom.view(batch_size, atom_max_num, atom_type)
            adj = batch_data.adj.view(batch_size, bond_type, atom_max_num, atom_max_num)
            x = x.to(device)
            adj = adj.to(device)
            adj_normalized = rescale_adj(adj).to(device)

            # Forward, backward and optimize
            z, sum_log_det_jacs = model(adj, x, adj_normalized)
            if config.multi_gpu:
                nll = model.module.log_prob(z, sum_log_det_jacs)
            else:
                nll = model.log_prob(z, sum_log_det_jacs)
            loss = nll[0] + nll[1]
            loss.backward()
            optimizer.step()
            tr.update()
            tr.print_summary()
            print('Nll_x:', nll[0].item(), 'Nll_adj:', nll[1].item())

        # Check out the training
        writer.add_scalar('Neg_log_like', loss.item(), epoch)
        writer.add_scalar('Neg_log_like_x', nll[0].item(), epoch)
        writer.add_scalar('Neg_log_like_adj', nll[1].item(), epoch)
        writer.add_scalar('Time(s/iter)', tr.get_avg_time_per_iter(), epoch)

        # Save the model checkpoints
        save_epochs = config.save_epochs
        if save_epochs == -1:
            save_epochs = config.max_epochs
        if (epoch + 1) % save_epochs == 0:
            os.makedirs(os.path.join(config.save_dir, time.ctime()[:-8]))
            if config.multi_gpu:
                torch.save(model.module.state_dict(), os.path.join(
                config.save_dir, time.ctime()[:-8], f'model_snapshot_epoch_{epoch + 1}'))
            else:
                torch.save(model.state_dict(), os.path.join(
                config.save_dir, time.ctime()[:-8], f'model_snapshot_epoch_{epoch + 1}'))
            tr.end()
    model_params.save(os.path.join(config.save_dir, time.ctime()[:-8], 'flow_params.json'))
    print("[Training Ends], Start at {}, End at {}".format(time.ctime(start), time.ctime()[:-8]))


if __name__ == '__main__':
    train()
