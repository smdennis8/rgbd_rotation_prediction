import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationNet(nn.Module):
    
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(RotationNet, self).__init__()
        self.layer1 = nn.Linear(8*64*64,4*32*32)
        self.layer2 = nn.Linear(4*32*32,2*16*16)
        self.layer3 = nn.Linear(2*16*16,8*8)
        self.layer4 = nn.Linear(8*8,4*4)
        self.layer5 = nn.Linear(4*4,9)
       
    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=8
        Out:
            output: Tensor [9]
        """
        model=nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5)
        output=model(x)
        
        #convert to valid rotation matrix
  
        return output


if __name__ == '__main__':
    model = RotationNet()
    input_tensor = torch.zeros([1, 8, 64, 64])
    output = model(input_tensor.reshape(1,-1))
    print("output size:", output.size())
    print(model)
