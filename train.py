import torch
import torch.nn as nn

from statistics import mean

def train(z_dim, batch_size, G, D, optimizer_G, optimizer_D, loader, device):
  
  ''' Set evaluation '''
  ones = torch.ones(batch_size).to(device)
  zeros = torch.zeros(batch_size).to(device)
  criterion = nn.MSELoss(reduction='mean')
  
  log_loss_G = []
  log_loss_D = []
  
  for _, (real_img, _) in enumerate(loader, 0):
    
    ''' (1) Train for Discriminator '''
    # Copy real images to GPU
    real_img = real_img.to(device)

    # The number of real images
    real_img_len = len(real_img)
    
    # Output real images in Discriminator
    real_out = D(real_img)
    
    # Calculate E[(D(x) - 1)^2]
    loss_D_real = criterion(real_out, ones[: real_img_len])
    
    # Set noize : z
    z = torch.randn(real_img_len, z_dim, 1, 1).to(device)
    fake_img_d = G(z)

    # Save fake images temporary
    fake_out = D(fake_img_d)
    
    # Calculate E[(D(G(z)))^2]
    loss_D_fake = criterion(fake_out, zeros[: real_img_len])
    
    # Sum two Discriminator's losses
    # 		E[(D(x) - 1)^2] + E[(D(G(z)))^2]
    loss_D = loss_D_real + loss_D_fake
    log_loss_D.append(loss_D.item())
    
    # BackPropagation
    D.zero_grad(), G.zero_grad()
    
    # Renew parameter
    loss_D.backward()
    optimizer_D.step()
    
    ''' (2) Train for Generator '''
    # Return fake images
    fake_img_g = G(z)
    
    # Output generated images in Discriminator
    out = D(fake_img_g)
    
    # Calculate evaluational function for generation model
    loss_G = criterion(out, ones[:real_img_len])
    log_loss_G.append(loss_G.item())
    
    # BackPropagation
    D.zero_grad(), G.zero_grad()
    
    # Renew parameter
    loss_G.backward()
    optimizer_G.step()
  
  return mean(log_loss_G), mean(log_loss_D)



