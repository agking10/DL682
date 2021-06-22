import torch
import torch.nn as nn

class SimpleUpsampler(nn.Module):
  def __init__(self):
    super(SimpleUpsampler, self).__init__()

    self.causal_conv = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding = 3),
        nn.ReLU(),
    )

    self.upsample = nn.ConvTranspose1d(64, 64, kernel_size=8, stride=2, padding=5)
    self.relu = nn.ReLU()

    self.final_conv = nn.Sequential(
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 1, kernel_size=5, padding=2),
        nn.Tanh()
    )

  def forward(self, x):
    out = self.causal_conv(x)
    out = self.relu(self.upsample(out))
    out = self.final_conv(out)
    return out