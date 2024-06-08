import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Helpers
def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d

# pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Functional
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward
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

# Patch-based image to token embedder
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        input_dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size[1] ==0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_size[0]//patch_size[0]) * (image_size[1]//patch_size[1])
        
        patch_dim = input_dim * patch_size[0] * patch_size[1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        
        # [B,N,C]
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # Add the positiombal embedding: same as the input: Learnable
        x += self.pos_embedding

        # With a dropout Layers
        return self.dropout(x)

# attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, tokens1, tokens2 = None, kv_include_self = False):
        '''Here Decide whether it is a self-attention or cross-attention.
        Based on 'kv_include_self'  '''
        b, n, _, h = *tokens1.shape, self.heads
        # Whether theres is a context, if not, Input is the context
        if tokens2 ==None:
            context = tokens1
        else:
            context = tokens2
        # include the cls tokens?
        if kv_include_self:
            context = tokens2 + tokens1

        qkv = (self.to_q(tokens1), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # q's shape is [B,nums_heads,nums_tokens,head_dimensions]
        # k's shape is [B,nums_head,nums_tokens,head_dimensions]
        # v's shape is [B,nums_head,nums_tokens,head_dimensions]
        
        # Get The Correlation Here
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        # Get The Output Here
        out = einsum('b h i j, b h j d -> b h i d', attn, v) # [B,nums_heads,1(cls),head_dimensions] --> Only Update the cls Tokens
    
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

# ProjectInOut
class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn # nn.Module
        need_projection = dim_in != dim_out
        # Make Sure When interactive : Two branch has same dimsion
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        # When feature fusion, keep the dimension same
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        # Feature Fusion over, recover the dimension
        x = self.project_out(x)
        return x

# Self-Attention Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads[_], dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# Cross-Attention Transformer
class CrossTransformer(nn.Module):
    def __init__(self, image1_dim, image2_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Convert the Image1 Tokens encoder to Image2 Tokens
                ProjectInOut(image1_dim, image2_dim, PreNorm(image2_dim, 
                                Attention(image2_dim, heads = heads[_], dim_head = dim_head, dropout = dropout))),
                
                # Convert the Image2 tokens encoder to Image1 one
                ProjectInOut(image2_dim, image1_dim, PreNorm(image1_dim, 
                                Attention(image1_dim, heads = heads[_], dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, feat1_tokens, feat2_tokens):
        
        for feat1_attend_feat2,feat2_attend_feat1 in self.layers:
            
            feat1_tokens = feat1_attend_feat2(feat1_tokens, tokens2 = feat2_tokens, kv_include_self = False) + feat1_tokens
            feat2_tokens = feat2_attend_feat1(feat2_tokens, tokens2 = feat1_tokens, kv_include_self = False) + feat2_tokens
        
        return feat1_tokens, feat2_tokens


# Transformer + Cross Transformer Basic Block
class BasicCrossVitBlock(nn.Module):
    def __init__(self,
                 depth=1,
                 embed_dim=[128,128],
                 image1_enc_params=None,
                 image2_enc_params =None,
                 cross_attn_heads =8,
                 cross_attn_depth = 3,
                 cross_attn_dim_head =64,
                 dropout = 0.
                 ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        # For Image One Self-Attention Transformer
        image1_enc_depths = image1_enc_params['depth']
        image1_enc_heads = image1_enc_params['heads']
        image1_mlp_dim = image1_enc_params['mlp_dim']
        image1_dim_head = image1_enc_params['dim_head']
        
        # For Image Two Self-Attention Transformer
        image2_enc_depths = image2_enc_params['depth']
        image2_enc_heads = image2_enc_params['heads']
        image2_mlp_dim = image2_enc_params['mlp_dim']
        image2_dim_head = image2_enc_params['dim_head']        
        
        # Cross Attention Blocks: SA1+ SA1 + CA12
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [ Transformer(dim=embed_dim[0],dropout=dropout,depth=image1_enc_depths,
                                  heads=image1_enc_heads,dim_head=image1_dim_head,
                                  mlp_dim=image1_mlp_dim),
                     Transformer(dim=embed_dim[1],depth=image2_enc_depths,
                                 heads=image2_enc_heads,dim_head=image2_dim_head,
                                 mlp_dim=image2_mlp_dim),
                      CrossTransformer(image1_dim=embed_dim[0],image2_dim=embed_dim[1],
                                       depth=cross_attn_depth,heads=cross_attn_heads,
                                       dim_head=cross_attn_dim_head,
                                       dropout=dropout)
                    ]))
            
    def forward(self,feat1_tokens,feat2_tokens):
        for feat1_enc, feat2_enc,cross_attend in self.layers:
            
            # Self-Attention Here
            feat1_tokens, feat2_tokens = feat1_enc(feat1_tokens),feat2_enc(feat2_tokens)
            
            # Cross Attention For feature fusion
            feat1_tokens, feat2_tokens = cross_attend(feat1_tokens,feat1_tokens)
            
        return feat1_tokens,feat1_tokens


# Cross-vit
class CrossVit(nn.Module):
    def __init__(self,
                 image_size=((40,80),(40,80)),
                 embedd_dim=(128,128),
                 input_dimension=(128,128),
                 patch_size=((1,1),(1,1)),
                 
                 basic_depth =1,
                 
                 cross_attention_depth =1,
                 cross_attention_head =[4],
                 cross_attention_dim_head =64,
                 
                 enc_depths =[3,3],
                 enc_heads=[[2,2,4],[2,2,4]],
                 enc_head_dim=[[64,64]],
                 enc_mlp_dims=[512,512],
                 
                 dropout_rate= 0.1,
                 emb_dropout=0.1,
                 skiped_patch_embedding=False
                 ):
        '''
        basic depth:              How many basic Blocks: (Transformer+Transformer+CrossTransfomer).
        cross_attention_head:     Heads used in Cross-Attention.
        enc  depths:              How many normal Transformer Blocks for each: Dim1 for first Image, Dim2 For second Image.
        enc_heads:                Heads use in normal transformer for each: Dim1 for first Image, Dim2 For second Image.
        enc_head_dim:             each head dimsion for each normal transformer:Dim1 for first Image, Dim2 For second Image.
        enc_mlp_dims:             Mlp for each head dimsion for each normal transformer: Dim1 for first Image, Dim2 For second Image.
        cross_attention_depth:    Each Basic Blocks, how many cross-attentions blocks are used.
        cross_attention_head:     Each basic blocks, for each attention head, how many attention heads are used.
        cross_attention_dim_head: for each cross-attention head, how many dimension does it contains.
        
        skip_patch_embedding:     Spilt into Patches With Positional Embedding.
        
        '''
        super(CrossVit,self).__init__()
        
        self.image_size = image_size
        self.input_dimension = input_dimension
        self.embedd_dim = embedd_dim
        self.patch_size = patch_size
        
        self.enc_depths = enc_depths
        self.basic_depth = basic_depth
        self.cross_attention_depth = cross_attention_depth
        self.cross_attention_head = cross_attention_head
        
        self.enc_heads = enc_heads
        self.enc_head_dim = enc_head_dim
        self.enc_mlp_dims = enc_mlp_dims
        
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.skiped_patch_embedding = skiped_patch_embedding
        
        if self.skiped_patch_embedding ==True:
            self.embedd_dim = self.input_dimension
        
        # Make Judgement
        assert self.embedd_dim[0]%self.enc_heads[0][0]==0
        assert self.embedd_dim[1]%self.enc_heads[1][0] ==0
        assert self.enc_depths[0] == len(self.enc_heads[0])
        assert self.enc_depths[1] == len(self.enc_heads[1])
        assert self.cross_attention_depth == len(self.cross_attention_head)

        
        if not self.skiped_patch_embedding:
            self.image1_embedder = ImageEmbedder(
                dim=self.embedd_dim[0],
                input_dim=self.input_dimension[0],
                image_size=self.image_size[0],
                patch_size=self.patch_size[0],
                dropout=emb_dropout
            )
            self.image2_embedder = ImageEmbedder(
                dim = self.embedd_dim[1],
                input_dim=self.input_dimension[1],
                image_size=self.image_size[1],
                patch_size=self.patch_size[1],
                dropout=emb_dropout
            )
            
            # Make Dictionary Here For Image1 and Image2
            image1_enc_params = dict(
                depth = enc_depths[0],
                heads = enc_heads[0],
                mlp_dim= enc_mlp_dims[0],
                dim_head = enc_head_dim[0]
            )
            image2_enc_params = dict(
                depth = enc_depths[1],
                heads = enc_heads[1],
                mlp_dim= enc_mlp_dims[1],
                dim_head = enc_head_dim[1]
            )
            
        self.basic_cross_transformer = BasicCrossVitBlock(depth=basic_depth,
                                                              embed_dim=self.embedd_dim,
                                                              image1_enc_params=image1_enc_params,
                                                              image2_enc_params= image2_enc_params,
                                                              cross_attn_depth=cross_attention_depth,
                                                              cross_attn_heads=self.cross_attention_head,
                                                              cross_attn_dim_head=cross_attention_dim_head,
                                                              dropout=dropout_rate)

    def forward(self,feat1,feat2):
    
        assert feat1.shape==feat2.shape
        B,C,H,W = feat1.shape
        
        after_height_patch = H//self.patch_size[0][0]
        after_width_patch= W//self.patch_size[0][1]
        
        # Spilt it into patches
        if not self.skiped_patch_embedding:
            feat1_tokens = self.image1_embedder(feat1)
            feat2_tokens = self.image2_embedder(feat2)
        else:
            feat1_tokens = feat1.flatten(2).permute(0,2,1).contiguous()
            feat2_tokens = feat2.flatten(2).permute(0,2,1).contiguous()
        
        # Make Cross Attention
        feat1_tokens, feat2_tokens = self.basic_cross_transformer(feat1_tokens,feat2_tokens)
        
        # feature fusion
        feat_fusion = feat1_tokens + feat2_tokens
        
        # Reshape To Image View
        feat_fusion = feat_fusion.permute(0,2,1).contiguous().view(B,-1,after_height_patch,after_width_patch)
        
        return feat_fusion
        


if __name__=="__main__":
    
    feature1 = torch.randn(1,24,40,80).cuda()
    feature2 = torch.randn(1,24,40,80).cuda()
    
    crossvit = CrossVit(image_size=[(40,80),(40,80)],
                        embedd_dim=[24,24],
                        input_dimension=(24,24),
                        patch_size=((1,1),(1,1)),
                        basic_depth=1,
                        cross_attention_dim_head=64,
                        cross_attention_depth=1,
                        cross_attention_head=[4],
                        enc_depths=[1,1],
                        enc_heads=[[4],[4]],
                        enc_head_dim=[64,64],
                        enc_mlp_dims=[128,128],
                        dropout_rate=0.1,
                        emb_dropout=0.1,
                        skiped_patch_embedding=False).cuda()
    
    feat_fusion = crossvit(feature1,feature2)
    
    print(feat_fusion.shape)