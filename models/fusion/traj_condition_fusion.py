import torch.nn as nn
import torch

class Mlp(nn.Module):
    """
    Multi-Layer Perceptron used in ViT-style blocks.
    Consists of two linear layers with GELU activation and dropout.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the MLP block.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    


class CrossAttention(nn.Module):
    """
    Cross Attention module.
    Attends to external query `y` over the key-value input `x`.
    """
    def __init__(self,
                 dim,   
                 q_dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(q_dim, dim, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        y = self.q(y)
        B1, N1, C1 = y.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_1 = self.qkv(y).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k_1.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v_1).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Traj_Hmap_Fusion_small(nn.Module):
    """
    Fusion module for trajectory heatmap and trajectory features.
    Uses linear projection + cross-attention to fuse initial heatmap with trajectory embedding.
    Outputs a flattened trajectory condition vector [B, 196].
    """
    def __init__(self, in_dim = 768, depth = 2):
        super(Traj_Hmap_Fusion_small, self).__init__()
        self.in_dim = in_dim
        self.depth = depth
        
        self.cross_attn = CrossAttention(in_dim, in_dim)
        
        # Flatten and project heatmap from [B, 32*32] to token [B, 768]
        self.linear = nn.Linear(32 * 32, 768)
        self.mlp = Mlp(in_features=in_dim, hidden_features=3 *in_dim, act_layer=nn.GELU, drop=0.)
        self.norm = nn.LayerNorm(in_dim)
        
        # Learnable positional embeddings
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 1, in_dim))
        
        # Final linear projection to 196-dim vector (14x14 token space)
        self.fc = nn.Linear(768, 196)
        self.relu = nn.ReLU()
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed2, std=0.02)
        
        # Initialize weights
        self.apply(_init_vit_weights)
        
    def forward(self, hmap_init, traj_feature):
        """
        Fuse an initial trajectory heatmap with trajectory-level feature via cross-attention.
        
        Args:
            hmap_init (Tensor): Initial 2D trajectory map, shape [B, 32, 32]
            traj_feature (Tensor): Trajectory semantic embedding, shape [B, 768]

        Returns:
            fused_condition (Tensor): Flattened fusion output [B, 196] used as CVAE condition
        """
        bs = hmap_init.shape[0]
        
        # Flatten and project heatmap to token representation
        traj_feature = traj_feature.unsqueeze(1) 
        x_in = hmap_init.reshape(bs, -1) 
        x_in = self.linear(x_in)
        x_in = self.relu(x_in)
        x_in = x_in.unsqueeze(1) 
        x_con = traj_feature 
        
        x_in = x_in + self.pos_embed1
        x_con = x_con + self.pos_embed2
        
        for i in range(self.depth):
            x_in = self.cross_attn(self.norm(x_in), self.norm(x_con))
            x_in= x_in + self.mlp(self.norm(x_in)) #[B, 1, 768]
            
        x_in = self.fc(x_in)
        x_in = self.relu(x_in)
        x_in = x_in.squeeze(1) #[B, 196]

        return x_in
    
def _init_vit_weights(m):
    """
    ViT-style weight initialization function.
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  
