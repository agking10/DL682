import torch.nn as nn

class LargeUpsampler(nn.Module):
  def __init__(self):
    super(LargeUpsampler, self).__init__()

    self.causal_conv = nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=13, padding=6),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=13, padding=6),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=13, padding=6),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Conv1d(64, 64, kernel_size=13, padding=7),
        nn.ReLU(),
    )

    self.upsample = nn.ConvTranspose1d(64, 64, kernel_size=8, stride=2, padding=5)
    self.relu = nn.ReLU()

    self.final_conv = nn.Sequential(
        nn.Conv1d(64, 64, kernel_size=13, padding=6),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=13, padding=6),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=13, padding=6),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 1, kernel_size=13, padding=6),
        nn.Tanh()
    )

  def forward(self, x):
    out = self.causal_conv(x)
    out = self.relu(self.upsample(out))
    out = self.final_conv(out)
    return out