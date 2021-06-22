import torch.nn as nn

class SimpleUpsampler3x(nn.Module):
  def __init__(self):
    super(SimpleUpsampler3x, self).__init__()

    self.causal_conv = nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=7, padding=3),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Conv1d(64, 64, kernel_size=9, padding=4),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Conv1d(64, 64, kernel_size=11, padding=5),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Conv1d(64, 128, kernel_size=13, padding = 6),
        nn.ReLU()
    )

    self.upsample = nn.ConvTranspose1d(128, 64, kernel_size=25, stride=3, padding=11)
    self.relu = nn.ReLU()

    self.final_conv = nn.Sequential(
        nn.Conv1d(64, 64, kernel_size=7, padding=3),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Conv1d(64, 64, kernel_size=9, padding=4),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Conv1d(64, 64, kernel_size=11, padding=5),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Conv1d(64, 1, kernel_size=13, padding=6),
        nn.Tanh()
    )

  def forward(self, x):
    out = self.causal_conv(x)
    out = self.relu(self.upsample(out))
    out = self.final_conv(out)
    return out