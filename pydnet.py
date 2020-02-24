import torch
import torch.nn as nn
from torch_layers import Conv2d_same_leaky, Convtranspose2d_same
import torch.nn.functional as F

class pyramid_feature(nn.Module):
    def __init__(self):
        super(pyramid_feature, self).__init__()
        # SCALE 6
        self.conv1a = Conv2d_same_leaky(in_channels=3,out_channels=16,kernel_size=3,stride=2)
        self.conv1b = Conv2d_same_leaky(in_channels=16,out_channels=16,kernel_size=3,stride=1)

        self.conv2a = Conv2d_same_leaky(in_channels=16,out_channels=32,kernel_size=3,stride=2)
        self.conv2b = Conv2d_same_leaky(in_channels=32,out_channels=32,kernel_size=3,stride=1)

        self.conv3a = Conv2d_same_leaky(in_channels=32,out_channels=64,kernel_size=3,stride=2)
        self.conv3b = Conv2d_same_leaky(in_channels=64,out_channels=64,kernel_size=3,stride=1)

        self.conv4a = Conv2d_same_leaky(in_channels=64,out_channels=96,kernel_size=3,stride=2)
        self.conv4b = Conv2d_same_leaky(in_channels=96,out_channels=96,kernel_size=3,stride=1)

        self.conv5a = Conv2d_same_leaky(in_channels=96,out_channels=128,kernel_size=3,stride=2)
        self.conv5b = Conv2d_same_leaky(in_channels=128,out_channels=128,kernel_size=3,stride=1)

        self.conv6a = Conv2d_same_leaky(in_channels=128,out_channels=192,kernel_size=3,stride=2)
        self.conv6b = Conv2d_same_leaky(in_channels=192,out_channels=192,kernel_size=3,stride=1)


    def forward(self, input):
        features = []
        
        # SCALE 1
        conv1a = conv1a(input)
        conv1b = conv1b(conv1a)
        features.append(conv1b)

        # SCALE 2
        conv2a = conv2a(input)
        conv2b = conv2b(conv2a)
        features.append(conv2b)

        # SCALE 3
        conv3a = conv3a(input)
        conv3b = conv3b(conv3a)
        features.append(conv3b)

        # SCALE 4
        conv4a = conv4a(input)
        conv4b = conv4b(conv4a)
        features.append(conv4b)

        # SCALE 5
        conv5a = conv5a(input)
        conv5b = conv5b(conv5a)
        features.append(conv5b)

        # SCALE 6
        conv6a = conv6a(input)
        conv6b = conv6b(conv6a)
        features.append(conv6b)

        return features
        
class estimater(nn.Module):
    def __init__(self, in_channels):
        super(estimater, self).__init__()
        
        self.conv_disp3 = Conv2d_same_leaky(in_channels, 96, 3)
        self.conv_disp4 = Conv2d_same_leaky(96, 64, 3)
        self.conv_disp5 = Conv2d_same_leaky(64, 32, 3)
        self.conv_disp6 = Conv2d_same_leaky(32, 8, 3,relu=False)

    def forward(self):
        disp3 = self.conv_disp3(self.disp2)
        disp4 = self.conv_disp4(disp3)
        disp5 = self.conv_disp4(disp4)
        disp6 = self.conv_disp6(disp5)

        return disp6


class get_disp(nn.Module):
    '''
    getting disparity
    '''
    def __init__(self, in_channels):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)


class pydnet(nn.Module):
    def __init__(self):
        super(pydnet, self).__init__()
        pyramid = pyramid_feature()
 
        # SCALE 6
        self.conv6 = estimater(192)
        self.disp7 = get_disp(8)

        upconv6 = nn.ConvTranspose2d(8,8,2,2)

        # SCALE 5
        self.conv5 = estimater(128 + 8)
        self.disp6 = get_disp(8)

        upconv5 = nn.ConvTranspose2d(8,8,2,2)

        # SCALE 4
        self.conv4 = estimater(96 + 8)
        self.disp5 = get_disp(8)

        upconv4 = nn.ConvTranspose2d(8,8,2,2)

        # SCALE 3
        self.conv3 = estimater(64 + 8)
        self.disp4 = get_disp(8)

        upconv3 = nn.ConvTranspose2d(8,8,2,2)

        # SCALE 2
        self.conv2 = estimater(32 + 8)
        self.disp3 = get_disp(8)

        upconv2 = nn.ConvTranspose2d(8,8,2,2)

        # SCALE 1
        self.conv1 = estimater(16 + 8)
        self.disp2 = get_disp(8)

        upconv1 = nn.ConvTranspose2d(8,8,2,2)

    def forward(self, x):
        pyramid = self.pyramid(x)

        # SCALE 6
        conv6 = self.conv6(pyramid[6])
        disp7 = self.disp7(conv6)
        upconv6 = self.upconv6(conv6)

        # SCALE 5
        map5 = torch.cat([pyramid[5],upconv6],1)
        conv5 = self.conv5(map5)
        disp6 = self.disp6(conv5)
        upconv5 = self.upconv5(conv5)

        # SCALE 4
        map4 = torch.cat([pyramid[4],upconv5],1)
        conv4 = self.conv4(map4)
        disp5 = self.disp5(conv4)
        upconv4 = self.upconv4(conv4)

        # SCALE 3
        map3 = torch.cat([pyramid[3],upconv4],1)
        conv3 = self.conv3(map3)
        disp4 = self.disp4(conv4)
        upconv3 = self.upconv6(conv3)

        # SCALE 2
        map2 = torch.cat([pyramid[2],upconv3],1)
        conv2 = self.conv2(map2)
        disp3 = self.disp3(conv3)
        upconv2 = self.upconv6(conv2)

        # SCALE 1
        map1 = torch.cat([pyramid[1],upconv2],1)
        conv1 = self.conv1(map1)
        disp2 = self.disp2(conv2)
        upconv1 = self.upconv6(conv1)

        