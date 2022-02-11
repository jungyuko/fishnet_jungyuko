import torch
import torch.nn as nn
from models.blocks import Residual_Block, UR_Block, DR_Block
import numpy as np

class Bridge(nn.Module): # Bridge
    def __init__(self, in_channel):
        super(Bridge, self).__init__()

        ############################################################################# 
        ## https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py #L45 ##
        self.bridge = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(in_channel, in_channel//16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//16, in_channel, kernel_size=1),
            
            nn.Sigmoid()
        )
        ############################################################################# 

    def forward(self, x):
        out = self.bridge(x)
        return out*x + x


class Score(nn.Module):
    def __init__(self, in_channel, out_channel=10, has_pool=False):
        super(Score, self).__init__()
        if has_pool == True:
            self.pred = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, in_channel//2, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel//2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel//2, out_channel, kernel_size=1, bias=True)
            )
        else:
            self.pred = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, in_channel//2, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel//2, out_channel, kernel_size=1, bias=True)
            )
            
    def forward(self, x):
        return self.pred(x)

###################################################################
## https://github.com/arabae/FishNet/blob/main/models/fishnet.py ##
class FishNet(nn.Module):
###################################################################
    def __init__(self,
        n_tail: int=3,
        n_body: int=3,
        n_head: int=3,
        in_channel: int=64,
        out_channel: int=64):
        super(FishNet, self).__init__()

        self.n_tail = n_tail
        self.n_body = n_body
        self.n_head = n_head
        
        self.in_channel  = in_channel
        self.out_channel = out_channel
        
        self.downsample = nn.MaxPool2d(2, stride=2)
        self.upsample   = nn.Upsample(scale_factor=2)


        ############################################################################
        # https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py #183 ##
        self.conv1 = self._conv_bn_relu(3, self.in_channel//2)
        self.conv2 = self._conv_bn_relu(self.in_channel//2, self.in_channel//2)
        self.conv3 = self._conv_bn_relu(self.in_channel//2, self.in_channel)
        self.pool1 = nn.MaxPool2d(3, padding=1, stride=2)
        ############################################################################
        self.FishTail()
        self.FishBody()
        self.FishHead()
        
        self.Score  = Score(self.channels[-1], 10, has_pool=True)
        self.bridge = Bridge(self.in_channel*2**(self.n_tail))

        
    ############################################################################
    # https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py #183 ## 
    def _conv_bn_relu(self, in_channel, out_channel, stride=1):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False),
                             nn.BatchNorm2d(out_channel),
                             nn.ReLU(inplace=True))
    ############################################################################

    def FishTail(self):
        n_tail = self.n_tail
        
        self.channels = [self.in_channel*2**(i) for i in range(n_tail+1)] # out channels

        self.tail_layer = nn.ModuleList([nn.Sequential(
            DR_Block(self.channels[i]),
            Residual_Block(self.channels[i]*2, self.channels[i]*2),
            self.downsample) for i in range(n_tail)])
        
    def FishBody(self):
        n_tail, n_body = self.n_tail, self.n_body

        self.tail_body_resolution = nn.ModuleList([Residual_Block(c, c) for c in self.channels[::-1]])

        for n in range(self.n_body):
            self.channels.append((self.channels[-1] + self.channels[n_tail-n])//2) 
        
        self.body_layer = nn.ModuleList([nn.Sequential(
            UR_Block(self.channels[i]*2),
            self.upsample) for i in range(n_tail+1, n_tail+n_body+1)])
    
    def FishHead(self):
        n_tail, n_body, n_head = self.n_tail, self.n_body, self.n_head

        self.channels.append(self.channels[-1] + self.channels[0]) 
        resolution_layers = [Residual_Block(self.channels[0], self.channels[0])]

        for n in range(self.n_head-1):
            self.channels.append(self.channels[-1] + self.channels[n_tail+n_body-n]*2)
            resolution_layers.append(Residual_Block(self.channels[n_tail+n_body-n]*2, self.channels[n_tail+n_body-n]*2))

        self.channels.append(self.channels[-1] + self.channels[n_tail])
        resolution_layers.append(Residual_Block(self.channels[n_tail], self.channels[n_tail]))

        self.body_head_resolution =  nn.ModuleList(resolution_layers)
        self.head_layer = nn.ModuleList([nn.Sequential(
            Residual_Block(self.channels[i], self.channels[i]),
            self.downsample) for i in range(n_tail+n_body+1, n_tail+n_body+n_head+1)])


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        
        tail_results = [x]
        body_inputs  = [x]
        
        ## tail
        for i in range(self.n_tail):
            x = self.tail_layer[i](x)
            tail_results.insert(0, x)

        x = self.bridge(x)
   
        first_body_input = x
        
        ## body
        for i in range(self.n_body):
            diff_module_same_resolution_feat = self.tail_body_resolution[i](tail_results[i])
            concat_feat = torch.cat((diff_module_same_resolution_feat, x), dim=1)  
            x           = self.body_layer[i](concat_feat)
            body_inputs.insert(1, concat_feat)
        
        ## head
        for i in range(self.n_head):
            diff_module_same_resolution_feat = self.body_head_resolution[i](body_inputs[i])
            concat_feat = torch.cat((diff_module_same_resolution_feat, x), dim=1)
            x           = self.head_layer[i](concat_feat)
            
        # calculate probability
        diff_module_same_resolution_feat = self.body_head_resolution[-1](first_body_input)
        concat_feat = torch.cat((diff_module_same_resolution_feat, x), dim=1)
        probs       = self.Score(concat_feat).squeeze(-1).squeeze(-1)
        
        return probs

def build_fishnet(args):
    return FishNet(
        n_tail=args.n_tail,
        n_body=args.n_body,
        n_head=args.n_head,
        in_channel=args.in_channel,
        out_channel=args.out_channel
    )