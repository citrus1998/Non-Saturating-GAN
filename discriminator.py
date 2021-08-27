import torch
import torch.nn as nn

class Discriminator(nn.Module) : 
    # gray => n_channel = 1, color => n_channel = 3
    def __init__(self, n_channel = 3, d_n_channel = 64) :
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(n_channel, d_n_channel * 1, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),

            nn.Conv2d(d_n_channel * 1, d_n_channel * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(d_n_channel * 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),

            nn.Conv2d(d_n_channel * 2, d_n_channel * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(d_n_channel * 4),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),

            nn.Conv2d(d_n_channel * 4, d_n_channel * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(d_n_channel * 8),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),

            nn.Conv2d(d_n_channel * 8, 1, kernel_size = 4, stride = 1, padding = 0)
        )
        
    def forward(self, x) :
        return self.layers(x).squeeze()
