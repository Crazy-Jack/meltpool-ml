import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np 
import pandas as pd 
import os
import argparse 
from tqdm import tqdm
import time 
import matplotlib.pyplot as plt 


parser = argparse.ArgumentParser(description='Type checking example')
parser.add_argument('--file', type=str, help='file to load')
parser.add_argument('--seed', type=int, default=1337, help='seed to generate the split')
parser.add_argument('--seed2', type=int, default=2043, help='seed to generate the split')
parser.add_argument('--train_split', type=float, default=0.8, help='portion of data that is used for training')
parser.add_argument('--batch_size', type=int, default=64, help='bz for training')
parser.add_argument('--block_size', type=int, default=3, help='block_size that used for the inference context length')
parser.add_argument('--max_iters', type=int, default=5000, help='iter to optimize for')
parser.add_argument('--eval_interval', type=int, default=10, help='')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--eval_iters', type=int, default=200, help='iteration number')
parser.add_argument('--n_embd', type=int, default=512, help='network latent embedding size')
parser.add_argument('--n_head', type=int, default=1, help='number of head')
parser.add_argument('--n_layer', type=int, default=3, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--embd_temp', type=float, default=1, help='temp for encoding input some dimension to large dimension')
parser.add_argument("--output_dir", type=str, default="output_baseline")

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
# read data
read_data_pd = pd.read_csv(args.file, skiprows=1)


# train/test split
raw_np = torch.tensor(read_data_pd.to_numpy(), dtype=torch.float32)
# raw_np = raw_np * 1000
# raw_np = (raw_np - raw_np.mean(0, keepdims=True)) / raw_np.std(0, keepdims=True)
raw_np = raw_np / raw_np.max(0)[0]
print(raw_np.shape)
print(f"raw_np max {raw_np[:, 3].max()} {raw_np[:, 4].max()}")
MAX_W = raw_np[:, 3].max()
MAX_D = raw_np[:, 4].max()
n = int(args.train_split * len(raw_np))
train_data = raw_np[:n]
print(train_data.shape)
val_data = raw_np[n:]
print(val_data.shape)

# exit(0)



class FlatDataset(Dataset):
    def __init__(self, data_np, block_size):
        data = data_np
        data_num = data.shape[0]
        ix = torch.arange(len(data_np) - block_size)
        x_control = torch.cat([data[i+block_size-1, 3:].unsqueeze(0) for i in ix], dim=0) # B,2
        x_context = torch.cat([data[i:i+block_size, :3].unsqueeze(0) for i in ix], dim=0) # B,T,3
        noise = torch.rand_like(x_context) * 0.03
        noise[:, :-1, :] = 0
        x_context += noise
        
        x_context = x_context.reshape(len(data_np) - block_size, -1) # B, T * 3
        self.x = torch.cat([x_control, x_context], dim=1) # B, 2 + T * 3
        self.y = torch.cat([data[i+block_size-1, :3].unsqueeze(0) for i in ix], dim=0) # B,3
        
    def __getitem__(self, idx):
       return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]

class FFTlayer(nn.Module):
    def __init__(self, seed, input_dim, n_embd, temp):
        super().__init__()
        torch.manual_seed(seed)
        
        self.register_buffer("B_matrix", torch.rand(input_dim, n_embd) * temp)
        
    
    def forward(self, x):
        x = torch.matmul(x, self.B_matrix) # B,T,n_embd
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x
        
        
train_set = FlatDataset(train_data, args.block_size)
val_set = FlatDataset(val_data, args.block_size)

train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

model = nn.Sequential(
            FFTlayer(args.seed, 2 + args.block_size * 3, args.n_embd, args.embd_temp),
            nn.ReLU(),
            nn.Linear(args.n_embd * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "width"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "depth"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)


epochs_loss = []

for epoch in range(1000 * 10):
    loss_epoch = 0.
    num_data = 0.
    for (data, target) in train_dataloader:
        data = data.to(device)
        target = target.to(device)
        
        logits = model(data)
        # loss = (((logits - target) ** 2)).mean()

        width = torch.cat([target[:, 0].unsqueeze(0), logits[:, 0].unsqueeze(0)], dim=0)
        depth = torch.cat([target[:, 1].unsqueeze(0), logits[:, 1].unsqueeze(0)], dim=0)
        corr0 = torch.corrcoef(width)[0, 1]
        corr1 = torch.corrcoef(depth)[0, 1]
        loss = 1 - (corr0 + corr1)/2
        
        loss += (((logits - target) ** 2)).mean()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_epoch += loss.item()
        num_data += data.shape[0]
    loss_epoch /= num_data
    epochs_loss.append(loss_epoch)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}], Loss: {loss.item()}')

    # eval
    if epoch % args.eval_interval == 0 or iter == args.max_iters - 1:
        
        # plot loss
        plt.clf()
        plt.plot(np.arange(len(epochs_loss)), epochs_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(args.output_dir, "loss.png"))
        
        
        with torch.no_grad():
            prediction = []
            targets = []
            for (data, target) in val_dataloader:
                data = data.to(device)
                target = target.to(device)
                targets.append(target)
                
                logits = model(data)
                prediction.append(logits)
            prediction = torch.cat(prediction, dim=0)
            
            targets = torch.cat(targets, dim=0)
            
            
            width = torch.cat([targets[:, 0].unsqueeze(0), prediction[:, 0].unsqueeze(0)], dim=0)
            depth = torch.cat([targets[:, 1].unsqueeze(0), prediction[:, 1].unsqueeze(0)], dim=0)
            
            # save
            if epoch % (args.eval_interval * 10) == 0 or iter == args.max_iters - 1:
                width_np = width.cpu().numpy()
                depth_np = depth.cpu().numpy()
                np.save(os.path.join(args.output_dir, "data", f"ep_{epoch}_width.npy"), width_np)
                np.save(os.path.join(args.output_dir, "data", f"ep_{epoch}_depth.npy"), depth_np)
                
            corr0 = torch.corrcoef(width)[0, 1].item()
            corr1 = torch.corrcoef(depth)[0, 1].item()
            width_acc = ((targets[:, 0].unsqueeze(0) - prediction[:, 0]).abs() / targets[:, 0].unsqueeze(0).abs()).mean().item()
            depth_acc = ((targets[:, 1].unsqueeze(0) - prediction[:, 1]).abs() / targets[:, 1].unsqueeze(0).abs()).mean().item()
            
            
            prediction = prediction.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            # prediction, targets 
            width_gt = targets[:, 0]
            width_pred = prediction[:, 0]
            index = np.argsort(width_gt)[::-1]
            width_gt = width_gt[index]
            width_pred = width_pred[index]
            
            plt.clf()
            plt.plot(np.arange(len(width_gt)), width_gt)
            plt.plot(np.arange(len(width_pred)), width_pred)
            plt.legend(["gt", "pred"])
            plt.xlabel("data test")
            plt.ylabel("normalized heat source")
            plt.title(f"corr {corr0:6f} error {width_acc:6f}")
            plt.ylim(0.55, 1)
            plt.savefig(os.path.join(args.output_dir, "width", f"ep{epoch}_width.jpg"))
            
            
            
            depth_gt = targets[:, 1]
            depth_pred = prediction[:, 1]
            index = np.argsort(depth_gt)[::-1]
            depth_gt = depth_gt[index]
            depth_pred = depth_pred[index]
            
            plt.clf()
            plt.plot(np.arange(len(depth_gt)), depth_gt)
            plt.plot(np.arange(len(depth_pred)), depth_pred)
            plt.xlabel("data test")
            plt.ylabel("normalized heat source")
            plt.legend(["gt", "pred"])
            plt.ylim(0.55, 1)
            plt.title(f"corr {corr1:6f} error {depth_acc:6f}")
            plt.savefig(os.path.join(args.output_dir, "depth", f"ep{epoch}_depth.jpg"))
            
            plt.clf()
            diff = np.abs(depth_pred - depth_gt) / depth_gt
            plt.plot(np.arange(len(depth_gt)), diff)
            plt.xlabel("data test")
            plt.ylabel("error")
            # plt.legend(["gt", "pred"])
            plt.ylim(0.0, 0.3)
            plt.title(f"error {depth_acc:6f}")
            plt.savefig(os.path.join(args.output_dir, "depth", f"ep{epoch}_differ_depth.jpg"))
            
            
            