import torch.nn as nn
import numpy as np
class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual_Block, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel//2, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),

            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.bottleneck(x)


class UR_Block(nn.Module): 
    def __init__(self, in_channel):
        
        super(UR_Block, self).__init__()
        self.k = 2

        self.M = Residual_Block(in_channel, out_channel=in_channel//2)

        ###############################################################################
        ## https://github.com/kevin-ssy/FishNet/blob/master/models/fish_block.py #67 ##
        def r(x): 
            batch, channel, height, width = x.size()
            x = x.view(batch, channel//self.k, self.k, height, width).sum(2)
            return x
        ###############################################################################

        self.r = r
    
    def forward(self, x):
        conv         = self.M(x)
        channel_wise = self.r(x)
        output       = channel_wise + conv

        return output

class DR_Block(nn.Module):
    def __init__(self, in_channel):

        super(DR_Block, self).__init__()

        self.M = Residual_Block(in_channel, out_channel=in_channel*2)

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel*2, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        conv   = self.M(x)
        res    = self.layer(x)
        output = res + conv

        return output
