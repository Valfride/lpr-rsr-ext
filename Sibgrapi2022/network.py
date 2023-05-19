import torch
import torch.nn as nn
# from torch.autograd import Variable
# from skimage.metrics import structural_similarity
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# import functions
# import numpy as np
# from PIL import Image

class AutoEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()        
        self.t = 3
        self.conv_in = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding='same', bias=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, self.t*in_channel, kernel_size=3, stride=1, groups=in_channel, padding='same', bias=True),
            nn.Conv2d(self.t*in_channel,         16, kernel_size=1, stride=1, padding='same', bias=True),            
            nn.PixelUnshuffle(2),
            nn.ReLU(),
            
            nn.Conv2d(64,        self.t*64, kernel_size=3, stride=1, groups=64, padding='same', bias=True),
            nn.Conv2d(self.t*64,        32, kernel_size=1, stride=1, padding='same', bias=True),
            nn.PixelUnshuffle(2),
            nn.ReLU(),
            )      
        
        self.decoder = nn.Sequential(
            
            nn.Conv2d(128       , self.t*128, kernel_size=3, stride=1, groups=128, padding='same', bias=True),
            nn.Conv2d(self.t*128, self.t*128*2**2, kernel_size=1, stride=1, padding='same', bias=True), 
            nn.PixelShuffle(2),
            nn.ReLU(),
            
            
            nn.Conv2d(self.t*128, self.t*64, kernel_size=3, stride=1, groups=self.t*64, padding='same', bias=True),
            nn.Conv2d(self.t*64, out_channel*2**2, kernel_size=1, stride=1, padding='same', bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding='same', bias=True),
            )
        
        self.GA = nn.Sequential(self.encoder,
                                self.decoder,
                                nn.ReLU()
                                )
        self.conv_out = nn.Conv2d(2*out_channel, out_channel, kernel_size=5, stride=1, padding='same', bias=True)
    def forward(self, x):
        out = self.GA(x)
        conv_in = self.conv_in(x)
        out = torch.cat((conv_in, out), dim=1)
        out = self.conv_out(out)
        return out
       
class TFAM(nn.Module):
    def __init__(self, channel_in_out, channel_mid_in = 128):
        super().__init__()
        self.convIn = nn.Sequential(
                      nn.Conv2d(channel_in_out, channel_in_out, kernel_size=3, stride=1, groups=channel_in_out, padding='same', bias=True),
                      nn.Conv2d(channel_in_out, channel_mid_in//16, kernel_size=1, stride=1, padding='same', bias=True)
                      )
        
        
        self.CA_unit = self.CA(channel_mid_in)
        self.POS_unit = self.POS(channel_mid_in)
        
        self.convEnd = nn.Sequential(
            nn.Conv2d(channel_mid_in+channel_mid_in//16, channel_mid_in+channel_mid_in//16, kernel_size=3, stride=1, groups=channel_mid_in//16, padding='same', bias=True),
                       nn.Conv2d(channel_mid_in+channel_mid_in//16, channel_in_out, kernel_size=1, stride=1, padding='same', bias=True)
                       )
        
    def CA(self, channel_mid_in):
        block = nn.Sequential(
            nn.PixelShuffle(1),
            nn.Conv2d(channel_mid_in//16, channel_mid_in, kernel_size=3, stride=1, groups=2, padding='same', bias=True)
        )
        
        return block
    
    def POS(self, channel_mid_in):
        block = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(channel_mid_in, channel_mid_in*4**2, kernel_size=7, stride=1, padding='same', bias=True),
            nn.PixelShuffle(4),
            nn.Conv2d(channel_mid_in, channel_mid_in+channel_mid_in//16, kernel_size=1, stride=1, padding='same', bias=True)
            )
        
        return block
    
    def forward(self, x):
        Input = self.convIn(x)
        ca = self.CA_unit(Input)
        ca = torch.cat((Input, ca), dim=1)
        pos = self.POS_unit(Input)
        out = torch.sigmoid(self.convEnd(torch.add(pos, ca)))
        out = torch.cat((out, x), dim=1)    

        return out
           
class AdaptiveResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.t = 3
        self.BNpath = nn.Sequential(
            nn.Conv2d(in_channels, self.t*in_channels, kernel_size=3, stride=1, groups=in_channels, padding='same', bias=True),
            nn.Conv2d(self.t*in_channels, 128, kernel_size=1, stride=1, padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(128, self.t*128, kernel_size=1, stride=1, groups=128, padding='same', bias=True),
            TFAM(self.t*128),
            nn.ReLU(),           
            nn.Conv2d(2*(self.t*128), in_channels, kernel_size=1, stride=1, padding='same', bias=True)
            )
        
        self.convMid = nn.Conv2d(in_channels, self.t*in_channels, kernel_size=3, groups=in_channels, stride=1, padding='same', bias=True)
        
        self.ADPpath = nn.Sequential(
            nn.PixelShuffle(1),
            nn.Conv2d(in_channels, self.t*in_channels, kernel_size=3, groups=in_channels, padding='same'),
            nn.Conv2d(self.t*in_channels, self.t*in_channels, kernel_size=1, padding='same'),
            nn.ReLU(),
            )
        
        self.convOUT = nn.Conv2d(self.t*in_channels, out_channels, kernel_size=1, stride=1, padding='same', bias=True)
        
        
    def forward(self, x):
        outBN = torch.add(self.BNpath(x), x)
        resPath = self.convMid(outBN)
        outADP = self.ADPpath(x)
        output = self.convOUT(torch.add(resPath, outADP))
        
        return output
    
class ResidualConcatenationBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ARB1 = AdaptiveResidualBlock(in_channels, 128)
        self.cat1 = in_channels + 128
        
        self.conv2 = nn.Conv2d(self.cat1, 128, kernel_size=1, stride=1, padding='same', bias=True)
        self.ARB2 = AdaptiveResidualBlock(128, 128)
        self.cat2 = self.cat1 + 128
        
        self.conv3 = nn.Conv2d(self.cat2, 128, kernel_size=1, stride=1, padding='same', bias=True)
        self.ARB3 = AdaptiveResidualBlock(128, 128)
        self.cat3 = self.cat2 + 128
        
        self.convOut = nn.Conv2d(self.cat3, out_channels, kernel_size=1, stride=1, padding='same', bias=True)
        
    def forward(self, x):
        ARB1 = torch.cat((self.ARB1(x), x), dim=1)
        
        conv2 = self.conv2(ARB1)
        ARB2 = torch.cat((self.ARB2(conv2), ARB1), dim=1) 
        
        conv3 = self.conv3(ARB2)
        ARB3 = torch.cat((self.ARB3(conv3), ARB2), dim=1)  
        
        output = self.convOut(ARB3)
        
        
        return output
  
class ResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.RCB1 = ResidualConcatenationBlock(in_channels, 256)
        self.cat1 = in_channels+256        
        self.conv2 = nn.Conv2d(self.cat1, 256, kernel_size=1, stride=1, padding='same', bias=True)
        self.RCB2 = ResidualConcatenationBlock(256, 256)
        self.cat2 = self.cat1+256        
        self.conv3 = nn.Conv2d(self.cat2, 256, kernel_size=1, stride=1, padding='same', bias=True)
        self.RCB3 = ResidualConcatenationBlock(256, 256)
        self.cat3 = self.cat2+256        
        self.convOUT = nn.Conv2d(self.cat3, out_channels, kernel_size=1, stride=1, padding='same', bias=True)
        
    def forward(self, x):
        RCB1 = torch.cat((self.RCB1(x), x), dim=1)        
        conv2 = self.conv2(RCB1)
        RCB2 = torch.cat((self.RCB2(conv2), RCB1), dim=1)        
        conv3 = self.conv3(RCB2)
        RCB3 = torch.cat((self.RCB3(conv3), RCB2), dim=1)
        
        output = self.convOUT(RCB3)
        
        return output
    
class FeatureModule(nn.Module):
    def __init__(self, channel_in, skip_connection_channels):
        super().__init__()
        self.TFAM = TFAM(channel_in)
        self.conv =  nn.Conv2d(2*channel_in, skip_connection_channels, kernel_size=3, stride=1, padding='same', bias=True)
        
    def forward(self, x, skip_connection):
        out = self.conv(self.TFAM(x))
        output = torch.add(out, skip_connection)
        
        return output

class Network(nn.Module):
    def __init__(self, in_channel, out_channels_input):
        super().__init__()
        self.inputLayer = AutoEncoder(in_channel, out_channels_input)
        self.RM = ResidualModule(out_channels_input, 128)
        self.FM = FeatureModule(128, out_channels_input)
        
        self.conv1 = nn.Conv2d(out_channels_input, 512*2**2, kernel_size=3, stride=1, padding='same', bias=True)
        self.PS1 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(512, 512*2**2, kernel_size=3, stride=1, padding='same', bias=True)
        self.PS2 = nn.PixelShuffle(2)
        
        self.out1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            )
            
        self.out2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            )
        
        self.out3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            )
        
        self.out4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            )
        
        self.out5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            )
        
        self.out6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            )

        self.out7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            )

        self.out = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Conv2d(512, 3, kernel_size=3, stride=1, padding='same', bias=True),
            nn.Sigmoid() 
            )
        
    def forward(self, x):
        Input = self.inputLayer(x)
        RMOutput = self.RM(Input)
        FMOutput = self.FM(RMOutput, Input)
        
        PS1 = self.conv1(FMOutput)
        PS1 = self.PS1(PS1)
        PS2 = self.conv2(PS1)
        out1 = self.out1(self.PS2(PS2))
        out2 = torch.add(self.out2(out1), out1)
        out3 = torch.add(self.out3(out2), out2)
        out4 = torch.add(self.out4(out3), out3)
        out5 = torch.add(self.out5(out4), out4)
        out6 = torch.add(self.out6(out5), out5)
        out7 = torch.add(self.out6(out6), out6)
        out = self.out(out7)
    
        return out
             
if __name__ == "__main__":
    feature_maps = torch.rand(3, 3, 120, 60)
    feature_maps1 = torch.rand(3, 10, 120, 60)
    '''Input layer debug'''
    # ILD = InputLayer(3, 6, 15)
    # print('InputLayer shapes', ILD(feature_maps).shape)
    
    '''TFAM DEBUG'''
    # tfam = TFAM(10)
    # tfam(feature_maps)
    # print('TFAM shape:', tfam(feature_maps).shape)
    
    '''AdaptiveResidualBlock DEBUG'''
    # ARB = AdaptiveResidualBlock(4, 16)
    # print('ARB shape:', ARB(feature_maps).shape)
    
    '''ResidualConcatenationBlock DEBUG'''
    # RCB = ResidualConcatenationBlock(4, 32)
    # print('RCB shape:', RCB(feature_maps).shape)
    
    '''ResidualModule DEBUG'''
    # RM = ResidualModule(4, 32)
    # print('ResidualModule shapes:', RM(feature_maps)[0].shape, RM(feature_maps)[1].shape)
    
    
    '''ResidualModule DEBUG'''
    # FM = FeatureModule(32, 8)
    # FM = FM(RM( RM(feature_maps)[1], feature_maps)[0])
    
    '''Network DEBUG'''
    net = Network(3, 42)
    # net(feature_maps).shape
    print('Network shapes', net(feature_maps).shape)
    
    # a = SSIMLoss()
    # print(a(feature_maps, feature_maps1))
    # inputLayer = AutoEncoder(45, (32, 64))
    # print(inputLayer(feature_maps).shape)
    # m = nn.AdaptiveAvgPool2d((64,32))
    # print(m(feature_maps).shape)
