import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
       
        #Down
        self.conv3_16 = nn.Conv2d(3,16,3, stride=1, padding=1)
        self.conv16_32 = nn.Conv2d(16,32,3, stride=1, padding=1)
        self.conv32_64 = nn.Conv2d(32,64,3, stride=1, padding=1)
        self.conv64_128 = nn.Conv2d(64,128,3, stride=1, padding=1)
        self.conv128_256 = nn.Conv2d(128,256,3, stride=1, padding=1)
        #Up
        self.conv128cat256_128 = nn.Conv2d(384,128,3, stride=1, padding=1)
        self.conv64cat128_64 = nn.Conv2d(192,64,3, stride=1, padding=1)
        self.conv32cat64_32 = nn.Conv2d(96,32,3, stride=1, padding=1)
        self.conv16cat32_16 = nn.Conv2d(48,16,3, stride=1, padding=1)
        #final Channel adjustment
        self.conv16_6 = nn.Conv2d(16,2,1, stride=1, padding=0)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        
        lay0across =  F.relu(self.conv3_16(x))
        x = F.max_pool2d(F.relu(self.conv3_16(x)), 2, stride=2)
        lay1across = F.relu(self.conv16_32(x))
        x = F.max_pool2d(F.relu(self.conv16_32(x)), 2, stride=2)
        lay2across = F.relu(self.conv32_64(x))
        x = F.max_pool2d(F.relu(self.conv32_64(x)), 2, stride=2)
        lay3across = F.relu(self.conv64_128(x))
        x = F.max_pool2d(F.relu(self.conv64_128(x)), 2, stride=2)
        
        x = F.relu(self.conv128_256(x))
     
        x = F.interpolate(x,scale_factor=2)
     
        x = F.interpolate(x,size=(30,40), mode='bicubic', align_corners=False)
        
        x = torch.cat((lay3across, x), 1)
        
        x = F.relu(self.conv128cat256_128(x))
       
        x = F.interpolate(x,size=(60,80), mode='bicubic', align_corners=False)

        x = torch.cat((lay2across, x), 1)
        
        x = F.relu(self.conv64cat128_64(x))
       
        x = F.interpolate(x,size=(120,160),mode='bicubic', align_corners=False)
       
        x = torch.cat((lay1across, x), 1)
        x = F.relu(self.conv32cat64_32(x))
       
        x = F.interpolate(x,size=(240,320), mode='bicubic', align_corners=False)

        x = torch.cat((lay0across, x), 1)

        x = F.relu(self.conv16cat32_16(x))
  
        x = F.relu(self.conv16_6(x))
                
        output = x
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
