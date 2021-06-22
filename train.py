import os
import numpy as np
import random
import torch.nn as nn
import importlib
import wave
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import torchaudio
import matplotlib.pyplot as plt
from timeit import default_timer as timer
# Change this to the model you want to train
from SimpleUpsampler import SimpleUpsampler

gpu_boole = torch.cuda.is_available()

#Function to write a tensor to a .wav file
#Input: tensor with shape [1, L]
def tensor2wav(fname, signal, sr):
  f = wave.open(fname, 'wb')
  f.setnchannels(1)
  f.setsampwidth(2)
  f.setframerate(sr)
  f.writeframes(signal.data.numpy().astype(np.uint16))
  f.close()


def collate_fn_chunk(batch):
    samples, targets = [], []
    for t in batch:
        samples.append(t['sample'])
        targets.append(t['target'])
    # concatenate along dimension 0 (chunk/minibatch)
    batch = {'sample': torch.cat(samples, dim=0), 'target': torch.cat(targets, dim=0)}
    return batch


# Function to evaluate the model
def model_eval(model, dataloader, loss_metric, mode="Train", verbose=True):
    model.eval()
    total_loss = 0
    n_batches = 0
    total_correct = 0
    total = 0

    for batch in dataloader:
        samples, targets = batch['sample'], batch['target']
        if gpu_boole:
            samples, targets = samples.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = model(samples)
            loss = loss_metric(outputs, targets)
            pred = torch.argmax(outputs, dim=1)
            N = samples.shape[0]
            total += N

        total_loss += loss.detach()
        n_batches += 1

    total_loss /= n_batches

    if verbose:
        print("Loss: %1.2f" % (total_loss.cpu().data.numpy().item()))

    return total_loss.cpu().data.numpy().item()


def load_data():
  data = torchaudio.datasets.LJSPEECH(root='.', download=True)
  return data

class testDataset(Dataset):
  def __init__(self,data_dir, sample_sr=12000, target_sr=24000 ):
    self.files = []
    self.data_dir = data_dir
    pathname=data_dir
    for filename in os.listdir(self.data_dir):
      self.files.append(os.path.join(pathname, filename))
    self.target_sr = target_sr
    self.sample_sr = sample_sr
    self.upsample = torchaudio.transforms.Resample(22050, target_sr)
    self.downsample = torchaudio.transforms.Resample(target_sr, sample_sr)
    self.ratio = target_sr // sample_sr
    self.mu_law = torchaudio.transforms.MuLawEncoding(quantization_channels=256)

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    #set_trace()
    target, _ = torchaudio.load(self.files[idx])
    target_sr = self.target_sr
    target = self.upsample(target)
    sample = self.transform_audio(target).squeeze(0)
    sample_sr = target_sr // self.ratio
    target = target.squeeze(0)

    # length of a chunk in seconds
    chunk_length_sec = 0.5
    # length of a chunk in number of samples
    chunk_length_sample = int(chunk_length_sec * sample_sr)
    chunk_length_target = int(chunk_length_sec * target_sr)
    # length of new audio, which must be divisible by the chunk length
    # We need to make sure the target sample is a multiple of our SR ratio or else
    # We won't be able to recover a sequence exactly the same length
    # Here I did not use integer division (or floor), but ceiling to keep all
    # samples and fill the rest of the tensor with zeros.
    new_length_sample = int(np.ceil(sample.size(0) / chunk_length_sample) * chunk_length_sample)
    new_length_target = int(np.ceil(target.size(0) / chunk_length_target) * chunk_length_target)

    # create new tensors
    _sample = torch.zeros([new_length_sample])
    _target = torch.zeros([new_length_target])
    _sample[:sample.size(0)] = sample
    _target[:target.size(0)] = target

    # convert dimensions to [num of chunks, 1, chunk length (fixed)]
    sample = _sample.view([-1, 1, chunk_length_sample])
    target = _target.view([-1, 1, chunk_length_target])

    return {'sample': sample, 'target': target}

  def transform_audio(self, signal):
    return self.downsample(signal.float())


# weight initialization
def weights_init(model):
  for i in model.modules():
    if isinstance(i, nn.Conv1d) or isinstance(i, nn.ConvTranspose1d):
      nn.init.xavier_uniform_(i.weight.data)

if not os.path.isdir('./LJSpeech-1.1/wavs'):
    load_data()

data_dir = './LJSpeech-1.1/wavs'
data = data = testDataset(data_dir, sample_sr=12000)

#Generate train, validation, test splits
N = len(data)
inds = range(len(data))
random.seed(0)
train_inds = random.sample(inds, int(0.6 * len(data)))
remaining = set(inds).difference(set(train_inds))
validation_inds = random.sample(remaining, int(0.5 * len(remaining)))
test_inds = remaining.difference(validation_inds)

train_data = Subset(data, train_inds)
val_data = Subset(data, validation_inds)
test_data = Subset(data, list(test_inds))

train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn_chunk)
val_loader = DataLoader(val_data, batch_size=2, collate_fn=collate_fn_chunk)
test_loader = DataLoader(test_data, batch_size=2, collate_fn=collate_fn_chunk)

# calling the network

model = SimpleUpsampler()
model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs = 10
loss_metric = nn.MSELoss()
if gpu_boole:
  model = model.cuda()

# parameters to calculate in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
# initial design = 101,249 param

loss_batch = []
loss_validation = []

for epoch in range(epochs):
    model.train()
    start = timer()

    for i, batch in enumerate(train_loader):
        samples, targets = batch['sample'], batch['target']
        # set_trace()
        if gpu_boole:
            samples, targets = samples.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(samples)
        loss = loss_metric(outputs, targets)
        loss.backward()
        optimizer.step()
        loss = loss.cpu().data.numpy().item()
        loss_batch.append(loss)

        del loss, outputs
        if i % 10 == 0:
            print(
                f'Epoch{epoch}: Time elapsed: {timer() - start:.2f} | Loss: {loss_batch[-1]:.5f} | Memory Allocated: {torch.cuda.memory_allocated()}')

    val_loss = model_eval(model, val_loader, loss_metric, mode="Validation")
    loss_validation.append(val_loss)

# Evaluate model on test set
model.eval()
base = torchaudio.transforms.Resample(12000, 24000)

base_loss = []
model_loss = []
N = 0

for batch in test_loader:
  samples, targets = batch['sample'], batch['target']
  with torch.no_grad():

    base_pred = base(samples)
    base_loss.append(loss_metric(base_pred, targets))
    samples, targets = samples.cuda(), targets.cuda()

    lite_pred = model(samples)
    N += 1
    model_loss.append(loss_metric(lite_pred, targets).cpu())

print(f'Loss: {np.array(model_loss).mean()} +/- {np.array(model_loss).std() / np.sqrt(N)}')
torch.save('model.torch', model)


