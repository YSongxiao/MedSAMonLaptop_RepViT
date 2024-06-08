import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
sys.path.append("../..")
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
#FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Attention_Absolute
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # FLC output
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Attention Relative
class AttentionRelative(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,
                 after_patch_height=None, after_patch_width = None
                 ):
        super().__init__()
        
        # Make Sure Tokens number is [after_patch_height*after_patch_weight]
        self.after_patch_height = after_patch_height
        self.after_patch_width = after_patch_width
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        #define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * after_patch_height - 1) * (2 * after_patch_width - 1), heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        
        # Get PairWise relative positional index for each token inside the window
        coords_h = torch.arange(self.after_patch_height)
        coords_w = torch.arange(self.after_patch_width)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.after_patch_height - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.after_patch_width - 1
        relative_coords[:, :, 0] *= 2 * self.after_patch_height - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # FLC output
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
    
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.after_patch_height*self.after_patch_width, self.after_patch_height * self.after_patch_width,-1)
        
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()
        dots = dots + relative_position_bias.unsqueeze(0)
        
        # Softmax operation Here
        attn = self.attend(dots)
        
        # Attention Dropout Here
        attn = self.dropout(attn)

        # Get Outputs
        out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer Blocks
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,
                 image_size=None,patch_size=None):
        super().__init__()
        
        after_patch_height = image_size[0]//patch_size[0]
        after_patch_width = image_size[1]//patch_size[1]
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionRelative(dim, heads = heads[_], dim_head = dim_head, dropout = dropout,
                                               after_patch_height=after_patch_height,after_patch_width=after_patch_width)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class ViT(nn.Module):
    def __init__(self,
                 image_size=(224,224),
                 patch_size=16,
                 embedd_dim = 512,
                 mlp_dim = 256,
                 depths = 3,
                 heads =[2,4,8],
                 input_channels=128,
                 dim_head = 64,
                 dropout_rate=0.,
                 emb_dropout =0.,
                 skiped_patch_embedding=False):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embedd_dim = embedd_dim
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.dropout_rate = dropout_rate
        self.depths  = depths
        self.heads = heads
        self.dim_head = dim_head
        self.skiped_patched_embedding = skiped_patch_embedding
        

        H,W = self.image_size
        patch_H, patch_W = self.patch_size
        
        assert H % patch_H == 0 and W % patch_W == 0, 'Image dimensions must be divisible by the patch size.'

        # Split the Input Image into patches
        num_patches = (H // patch_H) * (W // patch_W)
        # Each Patch's dimension Channels * patch_height * patch_width
        patch_dim = self.input_channels * patch_H * patch_W
        
        if not skiped_patch_embedding:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_H, p2 = patch_W),
                nn.Linear(patch_dim, self.embedd_dim),
        )
        
        # absolue postional encoding here
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.embedd_dim))
        
        
        self.dropout = nn.Dropout(self.emb_dropout)
        
        self.transformer = Transformer(self.embedd_dim, 
                                       self.depths, 
                                       self.heads, 
                                       self.dim_head, 
                                       self.mlp_dim, 
                                       self.dropout_rate,
                                       self.image_size,
                                       self.patch_size)        
    def forward(self,img):
        B,C,H,W = img.shape
        x = self.to_patch_embedding(img)
        b,n,_ = x.shape
        
        # Add absolute positional 
        x+= self.pos_embedding
        x = self.dropout(x)
        
        # Some transformer Blocks
        x = self.transformer(x)
        
        x = x.permute(0,2,1).view(B,-1,H//self.patch_size[0],W//self.patch_size[1])

        print(x.shape)

if __name__=="__main__":
    
    image = torch.randn(1,128,40,80).cuda()
    
    vit = ViT(image_size=(40,80),patch_size=(1,1),heads=(2,4,4),dim_head=64,depths=3,
              embedd_dim=512,mlp_dim=256,input_channels=128,dropout_rate=0.,emb_dropout=0.).cuda()
    
    vit(image)
    