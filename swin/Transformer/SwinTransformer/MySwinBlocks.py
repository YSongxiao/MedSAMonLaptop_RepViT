import sys
sys.path.append("../..")
import torch.nn as nn
import torch
import torch.nn.functional as F
from Transformer.SwinTransformer.PatchEmbedding import PatchEmbed
from Transformer.SwinTransformer.PatchMerging import PatchMerging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Transformer.SwinTransformer.BasicBlock import BasicLayer

class MySwinFormerBlocks(nn.Module):
    def __init__(self,
                 input_feature_size=(100,100),
                 input_feature_channels = 128,
                 window_size =7,
                 block_depths =[2,2,2],
                 nums_head=[4,4,4],
                 mlp_ratio = 4.0,
                 skiped_patch_embed=False,
                 patch_size=1,
                 embedd_dim=None,
                 use_ape=False,
                 use_prenorm=True,
                 norm_layer=None,
                 dropout_rate = 0.2,
                 downsample=False,
                 out_indices=(0,1,2),
                 frozen_stage=-1):
        '''
        input_feature_size(tuple or list) : The input Image shape, any shape supported by padding.
        input_feature_channels:       Input Feature's Channels Numbers, Make Sure can be divided by attention_heads.
        window_size:                  Swin-Former Local Window Size: Split into windows.
        blocls depths:                how many blocks is used: 2 is one(local_shift)
        nums_heads:                   For each blocks, how many heads are used.
        mlp_ratio:                    After FFN, the dimension increase ratio.
        patch size(tuple or list):    Tell how to downsample the input feature,default=1.
        embed_dim:          If spilt into batchs, each batch's hidden dimension.
        use_ape:            Whether use Absolute Positional Encoding.
        use_prenorm:        Whether use Pre Norm in the PatchEmbedding Phase.
        norm_layer:         If use Pre Norm, which normalization activation function will be used.
        dropout Rate:       the MLP dropout rate for regularization.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        frozen_stages:         How many layers that you do not want the network to learn
        '''
        super(MySwinFormerBlocks,self).__init__()
        self.input_feature_size = input_feature_size
        self.input_feature_channels = input_feature_channels
        self.window_size = window_size
        self.blocks_depths  = block_depths
        self.nums_head = nums_head
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.embed_dim = embedd_dim
        self.use_ape = use_ape
        self.norm_layer = norm_layer
        self.use_prenorm = use_prenorm
        self.skiped_patch_embed = skiped_patch_embed
        self.drop_out_rate = dropout_rate
        self.downsample = downsample
        self.qkv_bias = True
        self.qk_scale = None
        use_checkpoint=False
        self.attn_drop_rate = 0.
        self.drop_rate = 0
        self.out_indices = out_indices
        self.frozen_stages = frozen_stage

        self.pos_drop = nn.Dropout(p=self.drop_rate)
        # Make Sure the Multi-Head can be divided
        if isinstance(nums_head,list) or isinstance(nums_head,tuple):
            assert self.embed_dim%nums_head[0] ==0
        # Make Sure the output is equals to the inputs
        
        assert len(self.out_indices) == len(self.blocks_depths)
        assert len(self.blocks_depths) == len(self.nums_head)
        
        if self.skiped_patch_embed:
            try:
                assert self.patch_size[0]==1 or self.patch_size[1]==1
            except:
                print("Skiped Patch Embedding is only supported when the patch size =1!") 
                
        # split image into non-overlapping patches
        if not skiped_patch_embed:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size, in_chans=self.input_feature_channels, embed_dim=self.embed_dim,
                norm_layer=norm_layer if self.use_prenorm else None)
        
        # Add Absolute Positional Embedding?
                # absolute position embedding
        if self.use_ape:
            pretrain_img_size = self.input_feature_size
            patch_size = self.patch_size
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, patches_resolution[0], 
                                                               patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.num_layers = len(self.blocks_depths)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_out_rate, sum(self.blocks_depths))]  # stochastic depth decay rule
        
        if downsample:
            self.mlp_ratio = 4.0
            self.downsample_method = PatchMerging
        
        # build layers
        self.layers = nn.ModuleList()
        # How many layers, each layers has serveal swinformer Blocks
        for i_layer in range(self.num_layers):
            if downsample:
                dim = int(self.embed_dim * 2 ** i_layer)
            else:
                dim = int(self.embed_dim)
            layer = BasicLayer(
                dim=dim,
                depth=block_depths[i_layer],
                num_heads=nums_head[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(block_depths[:i_layer]):sum(block_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=self.downsample_method if ((i_layer < self.num_layers - 1) and self.downsample) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        if self.downsample:
            num_features = [int(self.embed_dim * 2 ** i) for i in range(self.num_layers)]
            self.num_features = num_features
        else:
            num_features = [int(self.embed_dim) for i in range(self.num_layers)]
            self.num_features = num_features
            

        # Add an norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        # Freeze Stages Parameters
        self._freeze_stages()
    
    # Freeze Stages
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        assert self.frozen_stages<=(self.num_layers+2)
        
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
        
        if not self.skiped_patch_embed:
            x = self.patch_embed(x)
        
        Wh,Ww = x.size(2),x.size(3)
        
        # Absolue Embedding (Optional)
        if self.use_ape:
            if self.absolute_pos_embed.size(-1)!=x.size(-1):
                absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            else:
                absolute_pos_embed = self.absolute_pos_embed
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2) # B Wh*Ww C
            
        # Enter the transformer Block : Window Attention and Shift Window
        '''
        Note the numbers of SwinFormer Blocks must be even number:
        odd for local windows,
        even for shift windows
        '''
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        
        return outs
        


if __name__=="__main__":
    input_feature = torch.randn(1,128,40,80).cuda()
    
    swinFormerBlocks = MySwinFormerBlocks(input_feature_size=(40,80),
                                          input_feature_channels=128,
                                          skiped_patch_embed=False,
                                          block_depths=[2,2,2],
                                          out_indices=(0,1,2),
                                          nums_head=[4,4,4],
                                          patch_size=(1,1),
                                          downsample=True,
                                          embedd_dim=128,
                                          use_ape=False,
                                          frozen_stage=-1,
                                          use_prenorm=True,
                                          norm_layer=nn.LayerNorm).cuda()

    output_feature = swinFormerBlocks(input_feature)
    
    for out in output_feature:
        print(out.shape)