import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

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

    def forward(self, x, context = None, kv_include_self = False):
        '''Here Decide whether it is a self-attention or cross-attention'''
        b, n, _, h = *x.shape, self.heads
        
        # Whether theres is a context, if not, Input is the context
        context = default(context, x)

        # include the cls tokens?
        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # q's shape is [B,nums_heads,1(cls),head_dimensions]
        # k's shape is [B,nums_head,nums_tokens,head_dimensions]
        # v's shape is [B,nums_head,nums_tokens,head_dimensions]
        
        # Get The Correlation Here
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Get The Output Here
        out = einsum('b h i j, b h j d -> b h i d', attn, v) # [B,nums_heads,1(cls),head_dimensions] --> Only Update the cls Tokens
    
        # Reshape: Multi-Head to one head: Still a CLS Layer
        out = rearrange(out, 'b h n d -> b n (h d)')

        
        return self.to_out(out)

# transformer encoder, for small and large patches
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
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

# cross attention transformer
class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 2 ProjectInOut
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Convert the smaller encoder to large one
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, 
                                Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                # Convert the large encoder to smaller one
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, 
                                Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        
        # Spilt The cls dimension and feature  dimension
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            # Here the context is the feature tokens: KV include itself is true
            # X is the small cls tokens
            
            # Update the small cls layers
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            # Update the large cls layers
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls
        # Concated Tokens
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        # Cross Attention Block: SA1+ SA2 + CA12
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            # Self attention For Each Feature Aggregation
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            
            # Cross Attention For Feature Fusion
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# patch-based image to token embedder

class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        
        # [B,N,C]
        x = self.to_patch_embedding(img)
    
        b, n, _ = x.shape

        # Get a cls token: Repeat at the batch dimension
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        # Concated the Cls Tokens into the INPIT as the first dimension
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add the positiombal embedding: same as the input: Learnable
        x += self.pos_embedding[:, :(n + 1)]

        # With a dropout Layers
        return self.dropout(x)

# cross ViT class
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 8,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, 
                                               image_size = image_size, 
                                               patch_size = sm_patch_size, 
                                               dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, 
                                               image_size = image_size, 
                                               patch_size = lg_patch_size, 
                                               dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # Get Smaller Tokens: (Split Batch + Cls Token + Positional Embedding)
        # [B,N,C]
        sm_tokens = self.sm_image_embedder(img)
        
        # Get large Tokens : (Split Batch + Cls Token + Positional Embedding)
        #[B,N,C]
        lg_tokens = self.lg_image_embedder(img)

        # print("Smaller Tokens: 1 + 224/8 * 224/8 = [1,785,128]:",sm_tokens.shape)
        # print("Larger Tokens: 1 + 224/16 * 224/16 =[1,197,128] :  ",lg_tokens.shape)
        
        
        # Multi-Scale Encoder
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # Get the smaller Tokens
        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        # Get smaller logits
        sm_logits = self.sm_mlp_head(sm_cls)
        
        # Get the big logits
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits
    


if __name__=="__main__":
    image = torch.randn(1,3,224,224).cuda()
    
    crossvit = CrossViT(image_size=224,num_classes=10,sm_dim=128,lg_dim=128).cuda()
    
    out = crossvit(image)
    print(out.shape)
