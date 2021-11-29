from torch import nn
import torch
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

#         self.fc1 = nn.Sequential(
#             nn.Linear(17, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 6)

#         )
        
        self.fc1 = nn.Sequential(
            nn.Linear(17, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)

        )
    def forward(self, x1):
        x2=self.fc1(x1)
        return x2