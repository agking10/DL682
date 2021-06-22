import torch.nn as nn

#Dilation block used in the wavenet and upsampling modules
class DilationBlock(nn.Module):
  def __init__(self, res_channels, skip_channels, dilation=1, kernel_size=25):
    super(DilationBlock, self).__init__()

    self.tanh_conv = nn.Conv1d(res_channels, 1,
                                  kernel_size=kernel_size, dilation=dilation,
                                  padding=dilation*(kernel_size - 1)//2)
    self.sig_conv = nn.Conv1d(res_channels, 1,
                                  kernel_size=kernel_size, dilation=dilation,
                                  padding=dilation*(kernel_size - 1)//2)
    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()
    self.output_conv = nn.Conv1d(1, res_channels, kernel_size=1)
    self.skip = nn.Conv1d(1, skip_channels, kernel_size=1)

  def forward(self, x):
    #set_trace()
    out1 = self.tanh(self.tanh_conv(x))
    out2 = self.sigmoid(self.sig_conv(x))
    out = out1 * out2
    skip = self.skip(out)
    res = self.output_conv(out)
    res = res + x[:,:,-out.shape[2]:]
    return res, skip