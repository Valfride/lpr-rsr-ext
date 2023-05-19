import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.metrics import structural_similarity
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import re
import cv2
import functions
import numpy as np
from PIL import Image

from keras.preprocessing.image import img_to_array

path_ocr = r'./saved_models/2023-02-02-exp-018-cn-paper-valfride-cg-ocr-goncalves2018realtime-original-120-40-adam-batch64-pat7'

class SSIMLoss(nn.Module):
    def __init__(self, OCR_path=True):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.SSIM_param = {
            'gaussian_weights':True,
            'multichannel':True,
            'sigma':1.5,
            'use_sample_covariance':False,
            'data_range':1.0
            }
        self.to_numpy = transforms.ToPILImage()
        if OCR_path == True:
            self.OCR_path = path_ocr
            self.OCR = functions.load_model(self.OCR_path)
            self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
            self.parameters = np.load(path_ocr + '/parameters.npy', allow_pickle=True).item()
            self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            self.padding = True
            self.aspect_ratio = self.IMAGE_DIMS[1]/self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
        else:
            self.OCR_path = False

    def compare_string(self, strA, strB):
        if (strA is None) or (strB is None):
            return 0
        
        size = min(len(strA), len(strB))
        counter = 0
        
        for i in range(size):
            if strA[i] == strB[i]:
                counter+=1
                
        return counter
    
    def OCR_pred(self, img, fl = None, convert_to_bgr=True):
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        if self.padding:
            img, _, _ = functions.padding(img, self.min_ratio, self.max_ratio, color = (127, 127, 127))        
        
        img = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
        img = img_to_array(img)        
        img = img.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
        
        img = (img/255.0).astype('float')
        predictions = self.OCR.predict(img)

        plates = [''] * 1
        for task_idx, pp in enumerate(predictions):
            idx_aux = task_idx + 1
            task = self.tasks[task_idx]

            if re.match(r'^char[1-9]$', task):
                for aux, p in enumerate(pp):
                    plates[aux] += self.ocr_classes['char{}'.format(idx_aux)][np.argmax(p)]
            else:
                raise Exception('unknown task \'{}\'!'.format(task))
                
        return plates
    
    def levenshtein(self, a, b):
        if not a: 
            return len(b)

        if not b: 
            return len(a)

        return min(self.levenshtein(a[1:], b[1:])+(a[0] != b[0]), self.levenshtein(a[1:], b)+1, self.levenshtein(a, b[1:])+1)
        
    def forward(self, SR, HR):     
        loss = self.criterion(SR, HR)

        return loss


class TFAM(nn.Module):
    def __init__(self, channel_in_out, channel_mid_in = 256):
        super().__init__()
        self.convIn = nn.Conv2d(channel_in_out, channel_mid_in, kernel_size=3,
                                stride=1, groups=1, padding='same', bias=True)
                      
        
        self.CA_unit = self.CA(channel_mid_in)
        self.POS_unit = self.POS(channel_mid_in)
        
        self.convEnd = nn.Conv2d(2*channel_mid_in, channel_in_out, kernel_size=3,
                                 stride=1, groups=1, padding='same', bias=True)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=7//2)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=7//2)
        
    def CA(self, channel_mid_in):
        block = nn.Sequential(
           nn.AvgPool2d(kernel_size=3, stride=1, padding=3//2),
           nn.Conv2d(channel_mid_in, channel_mid_in, kernel_size=1, stride=1,
                     groups=channel_mid_in, padding='same', bias=True)
        )
        
        return block
    
    def POS(self, channel_mid_in):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*channel_mid_in, 2*channel_mid_in, kernel_size=7, stride=2, groups=1, padding=7//2, bias=True)
        )
        
        return block
    
    def forward(self, x):
        Input = self.convIn(x)
        pos = torch.cat((self.avgpool(Input), self.maxpool(Input)), dim=1)
        pos = self.POS_unit(pos)
        ca = torch.cat((Input, self.CA_unit(Input)), dim=1)
        
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
            nn.LeakyReLU(),
            nn.Conv2d(128, self.t*128, kernel_size=1, stride=1, groups=128, padding='same', bias=True),
            TFAM(self.t*128),
            nn.LeakyReLU(),           
            nn.Conv2d(2*(self.t*128), in_channels, kernel_size=1, stride=1, padding='same', bias=True)
            )
        
        self.convMid = nn.Conv2d(in_channels, self.t*in_channels, kernel_size=3, groups=in_channels, stride=1, padding='same', bias=True)
        
        self.ADPpath = nn.Sequential(
            nn.PixelShuffle(1),
            nn.Conv2d(in_channels, self.t*in_channels, kernel_size=3, groups=in_channels, padding='same'),
            nn.Conv2d(self.t*in_channels, self.t*in_channels, kernel_size=1, padding='same'),
            nn.LeakyReLU(),
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
        self.convIn = nn.Conv2d(in_channel, out_channels_input, kernel_size=3,
                                stride=1, groups=1, padding='same', bias=True)
        self.RM = ResidualModule(out_channels_input, 128)
        self.FM = FeatureModule(128, out_channels_input)
        
        self.conv1 = nn.Conv2d(out_channels_input, 512*2**2, kernel_size=3, stride=1, padding='same', bias=True)
        self.PS1 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(512, 512*2**2, kernel_size=3, stride=1, padding='same', bias=True)
        self.PS2 = nn.PixelShuffle(2)
        
        self.convOut = nn.Conv2d(512, 3, kernel_size=3,
                                stride=1, groups=1, padding='same', bias=True)
        
    def forward(self, x):
        Input = self.convIn(x)
        RMOutput = self.RM(Input)
        FMOutput = self.FM(RMOutput, Input)
        
        PS1 = self.conv1(FMOutput)
        PS1 = self.PS1(PS1)
        PS2 = self.conv2(PS1)
        PS2 = self.PS2(PS2)
        out = self.convOut(PS2)
 
        return out
             
if __name__ == "__main__":
    feature_maps = torch.rand(3, 3, 40, 20)
    feature_maps1 = torch.rand(3, 3, 120, 60)
    '''Input layer debug'''
    # ILD = InputLayer(3, 6, 15)
    # print('InputLayer shapes', ILD(feature_maps).shape)
    
    '''TFAM DEBUG'''
    # tfam = TFAM(3)
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
    
    # a = SSIMLoss(OCR_path=False)
    # print(a(feature_maps, feature_maps1))
    # inputLayer = AutoEncoder(45, (32, 64))
    # print(inputLayer(feature_maps).shape)
    # m = nn.AdaptiveAvgPool2d((64,32))
    # print(m(feature_maps).shape)
