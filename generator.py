import torch
import torch.nn as nn

class Generator(nn.Module) :
    # gray => n_channel = 1, color => n_channel = 3
    def __init__(self, z_dim = 100, g_n_channel = 64, n_channel = 3) :
        super(Generator, self).__init__()
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_n_channel * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_n_channel * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(g_n_channel * 8, g_n_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_n_channel * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(g_n_channel * 4, g_n_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_n_channel * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(g_n_channel * 2, g_n_channel * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_n_channel * 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(g_n_channel * 1, n_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z) :
        return self.layers(z)
