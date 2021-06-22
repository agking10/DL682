import torch.nn as nn
from DilationBlock import DilationBlock


# Wavenet, as described in the paper by Google DeepMind
class Wavenet(nn.Module):

    def __init__(self, mu=256, n_residue=32, n_skip=512, dilations=10, repeat=1):
        super(Wavenet, self).__init__()

        self.dilations = [2 ** i for i in range(dilations)] * repeat
        self.causal_conv = nn.Conv1d(mu, n_residue, kernel_size=1)

        blocks = []
        tanh_conv = []
        sig_conv = []
        for x in self.dilations:
            blocks.append(DilationBlock(n_residue, n_skip, dilation=x, kernel_size=3))
        self.blocks = nn.ModuleList(blocks)

        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_skip, mu // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(mu // 2, mu, kernel_size=1)
        )
        self.dilation_blocks = nn.ModuleList(blocks)

        self.relu = nn.ReLU()

    def forward(self, x):
        # set_trace()
        res = self.relu(self.causal_conv(x))

        skip_connections = []
        i = 0
        for block in self.blocks:
            i += 1
            res, skip = block(res)
            skip_connections.append(skip)

        output = sum(skip_connections)
        output = self.output_layers(output)
        return output

#Upsampling network we can use to feed into the wavenet
class upsampler(nn.Module):
  def __init__(self, mu=256, n_blocks=5):
    super(upsampler, self).__init__();

    self.causal_conv = nn.Conv1d(1, 128, kernel_size=1)

    self.dilations = [2**i for i in range(n_blocks)]
    blocks = []
    for x in self.dilations:
      blocks.append(DilationBlock(res_channels=128, skip_channels=128, dilation=x, kernel_size=3))
    self.blocks = nn.ModuleList(blocks)

    self.upsample_x2 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2)
    self.upsample_x3 = nn.ConvTranspose1d(128, 1, kernel_size=3, stride=3)
    self.mid_conv = nn.Conv1d(128, 128, kernel_size=3, padding=1)
    self.down_conv = nn.Conv1d(128, 1, kernel_size=1)
    self.out_conv = nn.Conv1d(1, mu, kernel_size=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    res = self.causal_conv(x)
    skip_connections = []
    for block in self.blocks:
      res, skip = block(res)
      skip_connections.append(skip)
    out = self.relu(sum(skip_connections))
    out = self.relu(self.upsample_x2(out))
    for i in range(5):
      out = self.relu(self.mid_conv(out))
    out = self.relu(self.upsample_x3(out))
    out = self.out_conv(out)

    return out

class net(nn.Module):
  def __init__(self, mu=256, n_residue=32, n_skip=256, dilations=10, repeat=1, upsample_blocks=5):
    super(net, self).__init__()

    self.upsampler = upsampler(mu, upsample_blocks)
    self.wavenet = Wavenet(mu, n_residue, n_skip, dilations, repeat)

  def forward(self, x):
    h = self.upsampler(x)
    out = self.wavenet(h)
    return out