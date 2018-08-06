import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from continue_dataset import ShouldContinueDataset
import math
import models
from models import *
import os
import os.path
import numpy as np
import utils

"""
Train the network that predicts whether we should continue adding objects
"""
parser = argparse.ArgumentParser(description='Should Continue')
parser.add_argument('--data-dir', type=str, default="bedroom", metavar='S')
parser.add_argument('--num-workers', type=int, default=6, metavar='N')
parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
parser.add_argument('--train-size', type=int, default=6400, metavar='N')
parser.add_argument('--save-dir', type=str, default="train/bedroom", metavar='S')
parser.add_argument('--ablation', type=str, default=None, metavar='S')
parser.add_argument('--lr', type=float, default=0.001, metavar='N')
parser.add_argument('--eps', type=float, default=1e-6, metavar='N')
parser.add_argument('--use-count', action='store_true', default=False)
args = parser.parse_args()

save_dir = args.save_dir
utils.ensuredir(save_dir)
learning_rate = args.lr
batch_size = 16

with open(f"data/{args.data_dir}/final_categories_frequency", "r") as f:
    lines = f.readlines()
num_categories = len(lines)-2

if args.ablation is None:
    num_input_channels = num_categories+8
elif args.ablation == "basic":
    num_input_channels = 6
elif args.ablation == "depth":
    num_input_channels = 1
else:
    raise NotImplementedError

logfile = open(f"{save_dir}/log_continue.txt", 'w')
def LOG(msg):
    print(msg)
    logfile.write(msg + '\n')
    logfile.flush()

LOG('Building model...')
if args.use_count:
    model = models.resnet101(num_classes=2, num_input_channels=num_input_channels, use_fc=False)
    fc = FullyConnected(2048+num_categories, 2)
else:
    model = models.resnet101(num_classes=2, num_input_channels=num_categories+8)

loss = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

LOG('Converting to CUDA...')
model.cuda()
if args.use_count:
    fc.cuda()
loss.cuda()
softmax.cuda()

LOG('Building dataset...')
train_dataset = ShouldContinueDataset(
    data_root_dir = utils.get_data_root_dir(),
    data_dir = args.data_dir,
    scene_indices = (0, args.train_size),
    num_per_epoch = 1,
    complete_prob = 0.5,
    ablation = args.ablation
)

validation_dataset = ShouldContinueDataset(
    data_root_dir = utils.get_data_root_dir(),
    data_dir = args.data_dir,
    scene_indices = (args.train_size, args.train_size+160),
    num_per_epoch = 1,
    complete_prob = 0.5,
    seed = 42,
    ablation = args.ablation
)

LOG('Building data loader...')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    num_workers = args.num_workers,
    shuffle = True
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size = batch_size,
    num_workers = 0,
    shuffle = True
)

LOG('Building optimizer...')
optimizer = optim.Adam(list(model.parameters())+list(fc.parameters()),
    lr = learning_rate,
    betas = (0.9,0.999),
    eps = args.eps
)

if args.last_epoch < 0:
    load = False
    starting_epoch = 0
else:
    load = True
    last_epoch = args.last_epoch

if load:
    LOG('Loading saved models...')
    model.load_state_dict(torch.load(f"{save_dir}/continue_{last_epoch}.pt"))
    if args.use_count:
        fc.load_state_dict(torch.load(f"{save_dir}/continue_fc_{last_epoch}.pt"))
    optimizer.load_state_dict(torch.load(f"{save_dir}/continue_optim_backup.pt"))
    starting_epoch = last_epoch + 1

current_epoch = starting_epoch
num_seen = 0

model.train()
LOG(f'=========================== Epoch {current_epoch} ===========================')

def train():
    global num_seen, current_epoch
    for batch_idx, (data, target, existing) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        if args.use_count:
            existing = existing.cuda()
            existing = Variable(existing)

        optimizer.zero_grad()
        if args.use_count:
            o_conv = model(data)
            o_conv = torch.cat([o_conv, existing], 1)
            output = fc(o_conv)
        else:
            output = model(data)

        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()

        num_seen += batch_size
        if num_seen % 800 == 0:
            LOG(f'Examples {num_seen}/10000')
        if num_seen % 10000 == 0:
            LOG('Validating')
            validate()
            model.train()
            fc.train()
            num_seen = 0
            current_epoch += 1
            LOG(f'=========================== Epoch {current_epoch} ===========================')
            if current_epoch % 10 == 0:
                torch.save(model.state_dict(), f"{save_dir}/continue_{current_epoch}.pt")
                if args.use_count:
                    torch.save(fc.state_dict(), f"{save_dir}/continue_fc_{current_epoch}.pt")
                torch.save(optimizer.state_dict(), f"{save_dir}/continue_optim_backup.pt")

def validate():
    model.eval()
    fc.eval()
    total_loss = 0
    total_accuracy = 0
    for _, (data, target, existing) in enumerate(validation_loader):
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            if args.use_count:
                existing = existing.cuda()
                existing = Variable(existing)

            optimizer.zero_grad()

            if args.use_count:
                o_conv = model(data)
                o_conv = torch.cat([o_conv, existing], 1)
                output = fc(o_conv)
            else:
                output = model(data)

            loss_val = loss(output, target)
            total_loss += loss_val.cpu().data.numpy()

            output = softmax(output)
            outputs = output.cpu().data.numpy()
            targets = target.cpu().data.numpy()
            predictions = np.argmax(outputs, axis=1)
            num_correct = np.sum(predictions == targets)
            total_accuracy += num_correct / batch_size

    LOG(f'Loss: {total_loss/10}, Accuracy: {total_accuracy/10}')

while True:
    train()
