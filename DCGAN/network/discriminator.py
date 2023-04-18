from network.conv import ConvLR
from torch import nn
from torch import flatten
from torch.nn import Linear, LeakyReLU, Sigmoid

class Discriminator(nn.Module):
    def __init__(self, depth, alpha=0.2):
        super().__init__()
        self.conv1 = ConvLR(input_channels=depth, output_channels=32, k_size=4, stride=2, pad=1, a = alpha)
        self.conv2 = ConvLR(input_channels=32, output_channels=64, k_size=4, stride=2, pad=1, a = alpha)
        

        self.fc1 = Linear(in_features=3136, out_features=512)
        self.leakyRelu = LeakyReLU(alpha, inplace=True)
        
        self.fc2 = Linear(in_features=512, out_features=1)
        self.sigmoid = Sigmoid()
		
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.leakyRelu(x)

        x = self.fc2(x)
        output = self.sigmoid(x)
        
        return output