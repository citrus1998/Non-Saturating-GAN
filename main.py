import os
from IPython.display import Image, display_jpeg, display_png

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as data
from torch import optim 

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

z_dim = 100		# Noise
batch_size = 64
g_n_channel =32
d_n_channel = 32
n_epoch = 100

lr = 0.0002
betas = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

''' Set dataset we will use '''
img_data = ImageFolder("oxford-102/", 
                       transform=transforms.Compose([
                            transforms.Resize(80),
                            transforms.CenterCrop(64),
                            transforms.ToTensor()
                        ]))

img_loader = data.DataLoader(img_data, batch_size=batch_size, shuffle=True)

''' Save each directries '''
G_dir = './G_dir'
D_dir = './D_dir'
g_jpg = './g_jpg'

if not os.path.isdir(G_dir):
    os.makedirs(G_dir, exist_ok=True)

if not os.path.isdir(D_dir):
    os.makedirs(D_dir, exist_ok=True)

if not os.path.isdir(g_jpg):
    os.makedirs(g_jpg, exist_ok=True)

''' Set Generator (G) and Discriminator (D) '''
from generator import Generator
from discriminator import Discriminator

G = Generator(z_dim = z_dim, g_n_channel = g_n_channel).to(device)
D = Discriminator(d_n_channel = d_n_channel).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(betas, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(betas, 0.999))

fixed_z = torch.randn(batch_size, z_dim, 1, 1).to("cuda:0") # Set noise for the fixed Generator

''' Train '''
from train import train

l_Gs = []
l_Ds = []

for epoch in range(n_epoch) :

    l_G, l_D = train(z_dim, batch_size, G, D, optimizer_G, optimizer_D, img_loader, device)
    
    l_Gs.append(l_G)
    l_Ds.append(l_D)
    
    if epoch % 10 == 0 :
        torch.save(G.state_dict(), "G_dir/G_{:5d}.prm".format(epoch), pickle_protocol=4)
        torch.save(D.state_dict(), "D_dir/D_{:5d}.prm".format(epoch), pickle_protocol=4)
    
    z_image = G(fixed_z)
    save_image(z_image, "g_jpg/{:5d}.jpg".format(epoch))

''' Show a graph about the loss relationship between G and D '''
from graph import graph

graph(l_Gs, l_Ds)