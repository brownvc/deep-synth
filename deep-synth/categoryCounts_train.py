import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from categoryCounts_dataset import CategoryCountsDataset
from models.nade import DiscreteNADEModule
import math
import numpy as np
import utils

'''
Train the baseline NADE that samples object category counts for a scene
'''

parser = argparse.ArgumentParser(description='Occurence Baseline')
parser.add_argument('--data-dir', type=str, default="bedroom", metavar='S')
parser.add_argument('--num-workers', type=int, default=6, metavar='N')
parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
parser.add_argument('--train-size', type=int, default=6400, metavar='N')
parser.add_argument('--save-dir', type=str, default="train/bedroom", metavar='S')
parser.add_argument('--lr', type=float, default=0.001, metavar='N')
args = parser.parse_args()

save_dir = args.save_dir
utils.ensuredir(save_dir)

logfile = open(f"{save_dir}/log_count.txt", 'w')
def LOG(msg):
    print(msg)
    logfile.write(msg + '\n')
    logfile.flush()

start_epoch = 0
num_epochs = 50
learning_rate = args.lr
batch_size = 128

LOG('Building dataset...')
train_dataset = CategoryCountsDataset(
    data_root_dir = utils.get_data_root_dir(),
    data_dir = args.data_dir,
    scene_indices = (0, args.train_size),
)

LOG('Building data loader...')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    num_workers = args.num_workers,
    shuffle = True
)

LOG('Building model...')
data_size = train_dataset.data_size
data_domain_sizes = train_dataset.data_domain_sizes
model = DiscreteNADEModule(
    data_size = data_size,
    data_domain_sizes = data_domain_sizes,
    hidden_size = data_size
)

## TODO: Loading a checkpointed model would go here

LOG('Converting to CUDA...')
model.cuda()

LOG('Building optimizer...')
optimizer = optim.Adam(model.parameters(),
    lr = learning_rate,
    betas = (0.9,0.999),
    eps = 1e-6
)

current_epoch = 0
num_seen = 0
model.train()
LOG(f'=========================== Epoch {current_epoch} ===========================')

def train():
    global num_seen, current_epoch
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda()
        data = data.long()
        data = Variable(data)
        optimizer.zero_grad()
        lps = model(data)            # One log prob for each item in the batch
        loss = -torch.mean(lps)        # One, overall, average log prob for the whole batch
        loss.backward()
        optimizer.step()
        LOG(f'Batch {batch_idx}  |  Loss: {loss.cpu().data.numpy()}')

        num_seen += data.size()[0]
        if num_seen >= 10000:
            num_seen = 0
            current_epoch += 1
            LOG(f'=========================== Epoch {current_epoch} ===========================')
            if current_epoch % 5 == 0:
                torch.save(model.state_dict(), f"{save_dir}/categoryCounts_epoch_{current_epoch}.pt")
                torch.save(optimizer.state_dict(), f"{save_dir}/categoryCounts_optim_backup.pt")

while current_epoch < num_epochs:
    train()
