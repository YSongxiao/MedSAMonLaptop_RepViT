# --------------------------------------------------------
# Liu Zihua Transformer
# Copyright (c) 2022 Tokyo Institude of Technology
# Licensed under The MIT License [see LICENSE for details]
# Written by Liu Zihua
# --------------------------------------------------------
from string import printable
import sys
sys.path.append("../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from utils.devtools import print_tensor_shape
from collections import OrderedDict

from swin.Transformer.SwinTransformer.PatchEmbedding import PatchEmbed
from swin.Transformer.SwinTransformer.PatchMerging import PatchMerging
from swin.Transformer.SwinTransformer.BasicBlock import BasicLayer


class MySwinFormer(nn.Module):
    def __init__(self,
                 pretrain_image_size=(320,640),
                 patch_size=(4,4),
                 window_size=7,
                 frozen_stages=-1,
                 num_heads = [2,4,8,8],
                 depths =[2,2,6,2],
                 mlp_ratio =4,
                 qkv_bias = True,
                 drop_path_rate=0.2,
                 qk_scale=None,
                 in_chans=3,embed_dim=64,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 position_embed_dropout = 0,
                 drop_rate =0,
                 out_indices=(0, 1, 2, 3),
                 attn_drop_rate=0,
                 use_checkpoint=False,
                 if_absolute_embedding=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.if_absolute_embedding = if_absolute_embedding
        self.pretrain_image_size = pretrain_image_size
        self.num_heads = num_heads
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.qk_scale = qk_scale
        self.qkv_bias = qkv_bias
        self.out_indices = out_indices
        self.frozen_stages= frozen_stages
        
        # Spilt Patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size[0],in_chans=in_chans,
            embed_dim=embed_dim,norm_layer=norm_layer if patch_norm else None)
        

        # Absolue Embedding (Optional)
        if self.if_absolute_embedding:
            patches_resolution = [pretrain_image_size[0]//self.patch_size[0], pretrain_image_size[1]//self.patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1,self.embed_dim,patches_resolution[0],patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed,std=.02)
        
        self.pos_drop = nn.Dropout(p=position_embed_dropout)
        
        # Stochastic Depths: Dropout rate
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]
        self.num_layers = len(depths)
        # build layers: SwinFormer Layer
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            # if i_layer==self.num_features-2:
            #     print("dim: ",int(embed_dim * 2 ** i_layer))
            #     print("depth: ",depths[i_layer])
            #     print("num_heads: ",num_heads)
            #     print("window size: ",window_size)
            #     print("mlp_ratio: ",mlp_ratio)
                
            #     pass

            
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer) if i_layer<self.num_layers-1 else int(embed_dim * 2 ** (i_layer-1)) ,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 2) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        num_features[-1] = num_features[-2]
        self.num_features = num_features
        
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()
    
    # Freeze Stages
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # TODO
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
            
            
    def forward(self,x):
        
        '''Conv Spilt Patch'''
        x = self.patch_embed(x)
        # Cost Volume After patch's tokens
        Wh,Ww = x.size(2),x.size(3)
        
        # Absolute Positional Encoding
        if self.if_absolute_embedding:
            if self.absolute_pos_embed.size(-1)!=x.size(-1):
                absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            else:
                absolute_pos_embed = self.absolute_pos_embed
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)  # B Wh*Ww C
            
        
        x = self.pos_drop(x)
        
        
        # Window Attention
        outs = []
        for i in range(self.num_layers):
            
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)
        
        





if __name__=="__main__":
    input_tensor = torch.randn(1,3,256,256).cuda()
    
    swin_former = MySwinFormer(pretrain_image_size=(256,256),
                               patch_size=(1,1),in_chans=3,embed_dim=64,
                               norm_layer=nn.LayerNorm,
                               patch_norm=True,
                               if_absolute_embedding=True).cuda()
    
    outputs = swin_former(input_tensor)
    
    for out in outputs:
        print(out.shape)