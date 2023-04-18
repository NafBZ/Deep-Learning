from torch import nn


class ConvLR(nn.Module):
    def __init__(self, input_channels, output_channels, k_size, stride, pad, a):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=input_channels,
                                       out_channels=output_channels,
                                       kernel_size=k_size,
                                       stride=stride,
                                       padding=pad)
        
        self.activation = nn.LeakyReLU(alpha = a, inplace = True)
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x



class ConvTr(nn.Module):
    def __init__(self, input_channels, output_channels, k_size, stride, pad, b = False):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(in_channels=input_channels,
                                       out_channels=output_channels,
                                       kernel_size=k_size,
                                       stride=stride,
                                       padding=pad,
                                       bias=b)
        
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.activation = nn.ReLU()
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x