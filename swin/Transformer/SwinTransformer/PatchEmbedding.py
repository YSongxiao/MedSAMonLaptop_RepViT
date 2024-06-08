'''
Split Patches, Change Dimension, Add Layer Norm.
Focus Points: If input is no more rectangle, what it will do to process the input feature.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        
        #[4,4]
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Directly Use 4 by 4
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        
        # Padding Some
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        
        # Get 1/4 Size Feature Inputs
        x = self.proj(x)  # B C Wh Ww
        
        
        if self.norm is not None:
            # Number of tokens height, Numbers of tokens width
            Wh, Ww = x.size(2), x.size(3)
            # [B,C,N] --> [B,N,C]
            x = x.flatten(2).transpose(1, 2)
            # Layer Norm
            x = self.norm(x)
            # Change to feature shape: [B,C,H,W]
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


if __name__=="__main__":
    
    # Input Features
    input_feature = torch.randn(1,3,320,640).cuda()
    
    # Patch Embedding
    patch_embedding = PatchEmbed(patch_size=4,in_chans=3,embed_dim=96,norm_layer=nn.LayerNorm).cuda()
    
    # Get Results
    result = patch_embedding(input_feature)
    
    # print(result.shape)


