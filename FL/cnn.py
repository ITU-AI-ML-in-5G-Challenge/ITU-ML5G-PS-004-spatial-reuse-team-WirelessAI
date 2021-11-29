# !/usr/bin/python
# coding: utf8
# @Time    : 2018-08-05 19:22
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm

from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding =1),#若输入为100*100*1，滤波器为3*3*1，25个滤波器，输出size为98*98*25
            nn.BatchNorm2d(128),#让特征矩阵中的数值分布符合均值为0，方差为1的分布规律，网络会更好训练
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)#池化层，滤波器该组特征值中最大的那个值，输出Size:49*49*25
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, padding =1),#47*47*50 #256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)#23*23*50
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, padding = 1),#47*47*50 #256
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)#23*23*50
        )
        
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(512,1024, kernel_size=3, padding =1),#47*47*50 #256
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)#23*23*50
        )
        
        self.layer9 = nn.Sequential(
            nn.Conv2d(1024,2048, kernel_size=7, padding =1),#47*47*50 #256
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

#         self.layer10 = nn.Sequential(
#             nn.Conv2d(2048,4096, kernel_size=5, padding = 'same'),#47*47*50 #256
#             nn.BatchNorm2d(4096),
#             nn.ReLU(inplace=True)
#         )
        
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 17)
        )


    def forward(self, x1):
        x1= self.layer1(x1)
        x1= self.layer2(x1)
        x1= self.layer3(x1)
        x1= self.layer4(x1)
        x1= self.layer5(x1)
        x1= self.layer6(x1)
        x1= self.layer7(x1)
        x1= self.layer8(x1)
        x1= self.layer9(x1)
#         x1= self.layer10(x1)
#         x1= self.layer11(x1)
#         x1= self.layer12(x1)
        x1= self.gap(x1)
        x1 = x1.squeeze()
        x1=self.fc1(x1)
#         x1 = self.vgg(x1)
#         print(x1.shape)
        return x1








