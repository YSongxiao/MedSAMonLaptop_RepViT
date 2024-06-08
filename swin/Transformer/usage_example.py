
import torch
import torch.nn as nn
import sys
sys.path.append("../")
from Transformer.SwinTransformer.MySwinBlocks import MySwinFormerBlocks
from Transformer.VIT.vit_ape import ViT
from Transformer.CrossVit.crossvit_ape import CrossVit

'''
Usage Of SwinTransformerBlocks

'''

if __name__=="__main__":
    
    
    feature = torch.randn(3,128,40,80).cuda()
    
    # Example One: 
    # swinformer_blocks = MySwinFormerBlocks(input_feature_channels=128,
    #                                        window_size=7,
    #                                        embedd_dim=128,
    #                                        norm_layer=nn.LayerNorm,
    #                                        block_depths=[2,4],
    #                                        nums_head=[2,4],
    #                                        input_feature_size=(40,80),
    #                                        mlp_ratio=4.0,
    #                                        skiped_patch_embed=True,
    #                                        patch_size=(1,1),
    #                                        use_ape=True,
    #                                        use_prenorm=True,
    #                                        downsample=True,
    #                                        out_indices=(0,1),
    #                                        frozen_stage=-1).cuda()
    # out = swinformer_blocks(feature)
    # torch.Size([3, 128, 40, 80])
    # torch.Size([3, 256, 20, 40])
    
    
    # Example Two : Simple Block

    # swinformer_blocks = MySwinFormerBlocks(input_feature_channels=128,
    #                                        window_size=7,
    #                                        embedd_dim=128,
    #                                        norm_layer=nn.LayerNorm,
    #                                        block_depths=[2],
    #                                        nums_head=[2],
    #                                        input_feature_size=(40,80),
    #                                        mlp_ratio=4.0,
    #                                        skiped_patch_embed=True,
    #                                        patch_size=(1,1),
    #                                        use_ape=True,
    #                                        use_prenorm=True,
    #                                        downsample=False,
    #                                        out_indices=[0],
    #                                        frozen_stage=-1).cuda()
    # out = swinformer_blocks(feature)
    # for o in out:
    #     print(o.shape)
    
