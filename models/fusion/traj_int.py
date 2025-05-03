import torch.nn as nn
import torch

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, height=224, width = 224, patch_size=3, in_c=1, embed_dim = 1024, norm_layer=None):
        super().__init__()
        img_size = (height, width)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=16, stride=16)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    


class CrossAttention(nn.Module):
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
    
class Conv1dPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conv1dPooling, self).__init__()
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        # [b, 40, 768] -> [b, 768, 40]
        x = x.transpose(1, 2)
        output = self.linear(x)
        # [b, 768, 40] -> [b, 768]
        output = output.squeeze(-1)
        return output

class fusion_traj_int(nn.Module):
    def __init__(self, dim=768, input_num=32, con_num=1, depth=2):
        """
        Module to fuse trajectory-related text/image features using cross-attention.

        Args:
            dim (int): Embedding dimension.
            input_num (int): Number of tokens in input sequence.
            con_num (int): Number of conditioning tokens.
            depth (int): Number of cross-attention + MLP layers.
        """
        super(fusion_traj_int, self).__init__()
        self.con_num = con_num
        self.input_num = input_num
        self.depth = depth
        self.cross_attn = CrossAttention(dim, dim)
        self.pos_embed_input = nn.Parameter(torch.zeros(1, input_num, dim))
        self.pos_embed_condition = nn.Parameter(torch.zeros(1, con_num, dim))  
        self.norm = nn.LayerNorm(dim)  
        self.pool = Conv1dPooling(input_dim=768, output_dim=768)
        self.mlp = Mlp(in_features=dim, hidden_features=3 *dim, act_layer=nn.GELU, drop=0.)
        nn.init.trunc_normal_(self.pos_embed_input, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_condition, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x_input, x_condition):
        """
        Forward function for fusion.

        Args:
            x_input (Tensor): [B, 32, 768] - input feature sequence 
            x_condition (Tensor): [B, 1, 768] - single context vector 

        Returns:
            Tensor: [B, 768] - fused trajectory feature.
        """
        x_input = x_input + self.pos_embed_input
        x_condition = x_condition + self.pos_embed_condition
        for i in range(self.depth):
            x_input = self.cross_attn(self.norm(x_input), self.norm(x_condition))
            x_input= x_input + self.mlp(self.norm(x_input)) # Residual Connection
        # Pool along temporal dimension to get global feature
        x_input = self.pool(x_input)  # [B, 768]
        return x_input
    
    
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
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
        
        