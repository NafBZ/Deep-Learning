from torch import nn
from torch.nn import ConvTranspose2d, Tanh
from network.conv import ConvTr


class Generator(nn.Module):
	
    def __init__(self, inputNoise=100):
        super().__init__()

        self.conv1 = ConvTr(input_channels = inputNoise, output_channels = 128, k_size = 4, stride = 2, pad = 0)
        self.conv2 = ConvTr(input_channels = 128, output_channels = 64, k_size = 3, stride = 2, pad = 1)
        self.conv3 = ConvTr(input_channels = 64, output_channels = 32, k_size = 4, stride = 2, pad = 1)
        self.conv4 = ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.tanh = Tanh()
		

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.tanh(x)
        return output