import time
import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import StepLR
from config import get_config, get_sphere_config, Hyperparameters
from generate import load_model
from data.data_process import load_qm9_all
from data.three_dim_process import spherenet
from model import rescale_adj
from torch.utils.tensorboard import SummaryWriter

sphere_config = get_sphere_config()
snapshot_path = os.path.join(sphere_config.model_dir, sphere_config.snapshot_path)
params_path = os.path.join(sphere_config.model_dir, sphere_config.params_path)
model_params = Hyperparameters(path=params_path)

start = time.time()
print("Start at Time: {}".format(time.ctime()))
model_flow = load_model(snapshot_path, model_params, sphere_config.data_name)

if sphere_config.gpu >= 0:
    device = torch.device('cuda:' + str(sphere_config.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
model_flow.to(device)
model_flow.eval()

batch_size = sphere_config.batch_size
train_size = get_config().train_size
if sphere_config.data_name == 'qm9':
    train_loader, test_loader = load_qm9_all(train_size, batch_size, get_config().seed)
    atom_max_num = 9
    atom_type = 5
    bond_type = 4
    out_dim = 369

print('{} batchsize, iter {}'.format(
    batch_size,
    len(train_loader))
)

iter_per_epoch = len(train_loader)
max_epoch = sphere_config.max_epoch
learning_rate = sphere_config.learning_rate
lr_decay_factor = sphere_config.lr_decay_factor
lr_decay_step_size = sphere_config.lr_decay_step_size
save_interval = sphere_config.save_interval
model = spherenet(data_name=sphere_config.data_name, out_channels=out_dim)
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
writer = SummaryWriter()

for epoch in range(max_epoch):
    print(f'In epoch {epoch + 1}')
    for i, batch_data in enumerate(train_loader):
        if batch_data.atom.size()[0] != batch_size * atom_max_num:
            iter_per_epoch -= 1
            break
        optimizer.zero_grad()
        batch_data.to(device)
        x = batch_data.atom.view(batch_size, atom_max_num, atom_type)
        adj = batch_data.adj.view(batch_size, bond_type, atom_max_num, atom_max_num)
        adj_normalized = rescale_adj(adj)
        z, sum_log_det_jacs = model_flow(adj, x, adj_normalized)
        z0 = z[0].reshape(z[0].shape[0], -1)
        z1 = z[1].reshape(z[1].shape[0], -1)
        z_sum = torch.cat((z0, z1), 1)
        u = model(batch_data)
        loss = nn.MSELoss()
        output = loss(u, z_sum)
        output.backward()
        optimizer.step()

    writer.add_scalar('Alighment_loss', output.item(), epoch+1)
    print('loss:', output.item())
    scheduler.step()
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), os.path.join(sphere_config.model_dir, f'sphere_params_{epoch + 1}.pt'))

print('Saving checkpoint...')
checkpoint = {'epoch': max_epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
torch.save(checkpoint, os.path.join(sphere_config.model_dir, 'sphere_checkpoint.pt'))
