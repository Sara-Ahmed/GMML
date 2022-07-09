from torch.utils import *
from torch import nn
import torch,math
from torchvision.utils import *
from torchvision.transforms.functional import *

from utils import trunc_normal_


def sample_from_latent(mu,logvar):
    e = torch.randn_like(mu)
    sigma = (logvar/2).exp()
    return mu+ e*sigma

def compute_gaussian_kl(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    return 0.5 * torch.mean(
        torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1, [0])
    
def Block(out,kernel_size,stride):
    return nn.Sequential(
        nn.LazyConv2d(out,kernel_size,stride),nn.ReLU(),
        nn.BatchNorm2d(out),
        nn.Conv2d(out,out,1)
    )
    
def DeBlock(out,kernel_size,stride):
    return nn.Sequential(
        nn.LazyConvTranspose2d(out,kernel_size,stride),nn.ReLU(),
        nn.BatchNorm2d(out),
        nn.Conv2d(out,out,1)
    )
    
class VAERECHead(nn.Module):
    def __init__(self, in_dim, in_chans=3, patch_size=16):
        super().__init__()
        self.in_dim = in_dim

        layers = [nn.Linear(in_dim, in_dim)]
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.encoder = nn.Linear(in_dim,in_dim*2)
        
        self.convTrans = nn.ConvTranspose2d(in_dim, in_chans, 
                            kernel_size=(patch_size, patch_size), 
                            stride=(patch_size, patch_size))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def encode(self,x):
        mu, logvar = self.encoder(x).split([self.in_dim]*2,2)
        return mu, logvar

    def decode(self, x):
        x = self.mlp(x)
        
        x_rec = x.transpose(1, 2)
        out_sz = tuple( (  int(math.sqrt(x_rec.size()[2]))  ,   int(math.sqrt(x_rec.size()[2])) ) )
        x_rec = self.convTrans(x_rec.unflatten(2, out_sz))
                
        return x_rec
    
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = sample_from_latent(mu,logvar)
        x_rec = self.decode(z)
        
        return x_rec,z,(mu, logvar)
    
