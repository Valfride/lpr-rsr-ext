import torch
import torch.nn as nn
import collections

class DConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride=1, padding: str ='same', bias: bool =True):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = self.in_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = True
        
        self.dConv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, self.kernel_size, self.stride, groups=self.groups, padding=self.padding, bias=self.bias),
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)
            )
        
    def forward(self, x):
        x = self.dConv(x)
        return x
    
    
    
class invertedResBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int=3, stride: int=1, padding: str ='same', bias: bool =True, expansion: int = 3):
        super().__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = self.in_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = True
        
        self.expanded_features = expansion*self.in_channel
        
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channel, self.expanded_features, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.expanded_features),
            nn.ReLU(),
            DConv(self.expanded_features, self.expanded_features, self.kernel_size),
            nn.BatchNorm2d(self.expanded_features),
            nn.ReLU(),
            nn.Conv2d(self.expanded_features, self.out_channel, kernel_size=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
            )
        
        self.shortcut = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=self.stride)
        
    def forward(self, x):
        res = self.shortcut(x)
        x = self.block(x)
        out = x + res
        
        return x
    
class MobileNetV2(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int, stride=1, padding: str ='same', bias: bool =True, expansion: int = 3):
        super().__init__()
        
        self.blocks = nn.Sequential(
                    nn.Conv2d(in_channel, 32, kernel_size=1, stride=2, padding=0),
                    invertedResBlock(32, 16, 3),
                    
                    invertedResBlock(16, 24, 3, stride=2, padding=0),
                    invertedResBlock(24, 24, 3, stride=1, padding=0),
                    # # 
                    invertedResBlock(24, 32, 3, stride=2, padding=0),
                    invertedResBlock(32, 32, 3, stride=1, padding=0),
                    invertedResBlock(32, 32, 3, stride=1, padding=0),
                    
                    invertedResBlock(32, 64, 3, stride=2, padding=0),
                    invertedResBlock(64, 64, 3, stride=1, padding=0),
                    invertedResBlock(64, 64, 3, stride=1, padding=0),
                    invertedResBlock(64, 64, 3, stride=1, padding=0),
                    
                    invertedResBlock(64, 96, 3, stride=1),
                    invertedResBlock(96, 96, 3, stride=1),
                    invertedResBlock(96, 96, 3, stride=1),
                    
                    invertedResBlock(96, 160, 3, stride=2, padding=0),
                    invertedResBlock(160, 160, 3, stride=1, padding=0),
                    invertedResBlock(160, 160, 3, stride=1, padding=0),
                    
                    invertedResBlock(160, 1280, 3, stride=1, padding=0),
                    # nn.AvgPool2d(3),
                    nn.Conv2d(1280, 32, kernel_size=1, stride=1, padding='same'),
                    nn.Flatten(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                    nn.Sigmoid()
                )
            
        
    def forward(self, x):
        x = self.blocks(x)
        
        return x
        
if __name__ == "__main__":
    feature_maps = torch.rand(3, 3, 240, 120)
    
    net = MobileNetV2(3, 3)
    results = net(feature_maps)
    print(results.shape)
    
    
        # self.blocks = nn.Sequential(
        #     collections.OrderedDict(
        #             [
        #             ('InputConv', nn.Conv2d(in_channel, 32, kernel_size=1, stride=2, padding=0)),
        #             ('Bn_stride1-0'),  invertedResBlock(32, 16, 3),
                    
        #             # ('Bn_stride2-0'),  invertedResBlock(16, 24, 3, stride=2, padding=0),
        #             # ('Bn_stride2-1'),  invertedResBlock(24, 24, 3, stride=2, padding=0),
        #             # # 
        #             # ('Bn_stride2-2'),  invertedResBlock(24, 32, 3, stride=2, padding=0),
        #             # ('Bn_stride2-3'),  invertedResBlock(32, 32, 3, stride=2, padding=0),
        #             # ('Bn_stride2-4'),  invertedResBlock(32, 32, 3, stride=2, padding=0),
                    
        #             # ('Bn_stride2-5'),  invertedResBlock(32, 64, 3, stride=2, padding=0),
        #             # ('Bn_stride2-6'),  invertedResBlock(64, 64, 3, stride=2, padding=0),
        #             # ('Bn_stride2-7'),  invertedResBlock(64, 64, 3, stride=2, padding=0),
        #             # ('Bn_stride2-8'),  invertedResBlock(64, 64, 3, stride=2, padding=0),
                    
        #             # ('Bn_stride1-1'),  invertedResBlock(64, 96, 3, stride=1),
        #             # ('Bn_stride1-2'),  invertedResBlock(96, 96, 3, stride=1),
        #             # ('Bn_stride1-3'),  invertedResBlock(96, 96, 3, stride=1),
                    
        #             # ('Bn_stride2-9'),  invertedResBlock(96, 160, 3, stride=2, padding=0),
        #             # ('Bn_stride2-10'),  invertedResBlock(160, 160, 3, stride=2, padding=0),
        #             # ('Bn_stride2-11'),  invertedResBlock(160, 160, 3, stride=2, padding=0),
        #             ]   
                 
        #         )
        #     )
