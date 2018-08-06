import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import *
from location_dataset import LocationDataset
import numpy as np
import math
import utils

parser = argparse.ArgumentParser(description='Location Training with Auxillary Tasks')
parser.add_argument('--data-dir', type=str, default="bedroom", metavar='S')
parser.add_argument('--num-workers', type=int, default=6, metavar='N')
parser.add_argument('--last-epoch', type=int, default=-1, metavar='N') #If positive, use saved epoch
parser.add_argument('--train-size', type=int, default=6400, metavar='N')
parser.add_argument('--save-dir', type=str, default="train/bedroom", metavar='S')
parser.add_argument('--ablation', type=str, default=None, metavar='S')
parser.add_argument('--lr', type=float, default=0.001, metavar='N')
parser.add_argument('--eps', type=float, default=1e-6, metavar='N')
parser.add_argument('--p-auxiliary', type=float, default=0.0, metavar='N')
parser.add_argument('--use-count', action='store_true', default=False) #Use category count if true
parser.add_argument('--no-penalty', action='store_true', default=False) #If True, L_Global is not used
parser.add_argument('--progressive-p', action='store_true', default=False) #If True, start with lower p_auxiliary and gradually increase
args = parser.parse_args()

save_dir = args.save_dir
utils.ensuredir(save_dir)
batch_size = 16

with open(f"data/{args.data_dir}/final_categories_frequency", "r") as f:
    lines = f.readlines()
num_categories = len(lines)-2

if args.ablation is None:
    num_input_channels = num_categories+9
elif args.ablation == "basic":
    num_input_channels = 7
elif args.ablation == "depth":
    num_input_channels = 2
else:
    raise NotImplementedError

logfile = open(f"{save_dir}/log_location.txt", 'w')
def LOG(msg):
    print(msg)
    logfile.write(msg + '\n')
    logfile.flush()

LOG('Building model...')
model = resnet101(num_classes=num_categories+3, num_input_channels=num_input_channels, use_fc=False)
if args.use_count:
    fc = FullyConnected(2048 + num_categories, num_categories+3)
else:
    fc = FullyConnected(2048, num_categories+3)

cross_entropy = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

LOG('Converting to CUDA...')
model.cuda()
fc.cuda()
cross_entropy.cuda()
softmax.cuda()

LOG('Building dataset...')
train_dataset = LocationDataset(
    data_root_dir = utils.get_data_root_dir(),
    data_dir = args.data_dir,
    scene_indices = (0, args.train_size),
    p_auxiliary = args.p_auxiliary,
    ablation = args.ablation
)
#Size of validation set is 160 by default
validation_dataset = LocationDataset(
    data_root_dir = utils.get_data_root_dir(),
    data_dir = args.data_dir,
    scene_indices = (args.train_size, args.train_size+160),
    seed = 42,
    p_auxiliary = 0, #Only tests positive examples in validation
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
    lr = args.lr,
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
    model.load_state_dict(torch.load(f"{save_dir}/location_{last_epoch}.pt"))
    fc.load_state_dict(torch.load(f"{save_dir}/location_fc_{last_epoch}.pt"))
    optimizer.load_state_dict(torch.load(f"{save_dir}/location_optim_backup.pt"))
    starting_epoch = last_epoch + 1

current_epoch = starting_epoch
num_seen = 0

if args.progressive_p:
    if current_epoch <= 30:
        train_dataset.p_auxiliary = 0.0
    if current_epoch > 30:
        train_dataset.p_auxiliary = 0.5
    if current_epoch > 60:
        train_dataset.p_auxiliary = 0.7
    if current_epoch > 90:
        train_dataset.p_auxiliary = 0.9
    if current_epoch > 120:
        train_dataset.p_auxiliary = 0.95

model.train()
LOG(f'=========================== Epoch {current_epoch} ===========================')

def train():
    global num_seen, current_epoch
    for batch_idx, (data, t, existing, penalty) \
                   in enumerate(train_loader):
        
        data, t = data.cuda(), t.cuda()
        existing, penalty = existing.cuda(), penalty.cuda()
        
        optimizer.zero_grad()
        o_conv = model(data)
        
        if args.use_count:
            o_conv = torch.cat([o_conv, existing], 1)
    
        o = fc(o_conv)
        loss = cross_entropy(o,t)
        
        if not args.no_penalty:
            o_s = softmax(o)[:,0:num_categories]
            l_penalty = (o_s * penalty).sum()

        if not args.no_penalty:
            loss += l_penalty
        loss.backward()
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
            LOG(f'{train_dataset.p_auxiliary}')
            if current_epoch % 5 == 0:
                torch.save(model.state_dict(), f"{save_dir}/location_{current_epoch}.pt")
                torch.save(fc.state_dict(), f"{save_dir}/location_fc_{current_epoch}.pt")
                torch.save(optimizer.state_dict(), f"{save_dir}/location_optim_backup.pt")

            if args.progressive_p:
                if current_epoch <= 30:
                    train_dataset.p_auxiliary = 0.0
                if current_epoch > 30:
                    train_dataset.p_auxiliary = 0.5
                if current_epoch > 60:
                    train_dataset.p_auxiliary = 0.7
                if current_epoch > 90:
                    train_dataset.p_auxiliary = 0.9
                if current_epoch > 120:
                    train_dataset.p_auxiliary = 0.95

def validate():
    model.eval()
    fc.eval()
    total_loss = 0
    total_accuracy = 0
    for batch_idx, (data, t, existing, penalty) \
                   in enumerate(validation_loader):

        with torch.no_grad():
            data, t = data.cuda(), t.cuda()
            existing = existing.cuda()

            optimizer.zero_grad()
            o_conv = model(data)
            if args.use_count:
                o_conv = torch.cat([o_conv, existing], 1)
            o = fc(o_conv)
            l = cross_entropy(o,t)
            total_loss += l.cpu().data.numpy()
            
            output = softmax(o)
            outputs = output.cpu().data.numpy()
            targets = t.cpu().data.numpy()
            predictions = np.argmax(outputs, axis=1)
            num_correct = np.sum(predictions == targets)
            total_accuracy += num_correct / batch_size

    LOG(f'Loss: {total_loss/10}, Accuracy: {total_accuracy/10}')

while True:
    train()
