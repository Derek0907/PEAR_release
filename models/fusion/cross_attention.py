import torch.nn as nn
import torch
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, height=224, width = 224, patch_size=3, in_c=1, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (height, width)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=7, stride=4 ,padding=3)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.apply(_init_weights)

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
        self.apply(_init_weights)

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
        self.apply(_init_weights)

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
    
    
class Cross_img(nn.Module):
    def __init__(self, dim=768, num_img=3136, num_text=32, depth=2):
        super(Cross_img, self).__init__()
        self.num_img = num_img
        self.num_text = num_text
        self.depth = depth
        self.dim = dim

        # Convert image [1, 224, 224] into patch embeddings [B, 3136, dim]
        self.patch_embed = PatchEmbed(in_c=1, embed_dim=dim)

        # Cross attention module to align image with text
        self.cross_attn = CrossAttention(dim, dim)

        # Positional embeddings for image and text tokens
        self.pos_embed_input = nn.Parameter(torch.zeros(1, num_img, dim))
        self.pos_embed_condition = nn.Parameter(torch.zeros(1, num_text, dim))  

        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=3 * dim, act_layer=nn.GELU, drop=0.)

        # Output projection to single channel (e.g., heatmap)
        self.fc = nn.Linear(dim, 1)
        self.relu = nn.ReLU()

        nn.init.trunc_normal_(self.pos_embed_input, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_condition, std=0.02)
        self.apply(_init_weights)
        
    def forward(self, img, text):
        '''
        img: tensor of shape [B, 1, 224, 224] — image input
        text: tensor of shape [B, 32, 768] — text embedding 
        '''
        bs = img.shape[0]

        # Convert image to patch tokens
        x_in = self.patch_embed(img)  # [B, 3136, 768]
        x_con = text  # [B, 32, 768]

        # Add learnable positional embeddings
        x_in = x_in + self.pos_embed_input
        x_con = x_con + self.pos_embed_condition

        # Apply multi-layer cross-attention
        for _ in range(self.depth):
            x_in = self.cross_attn(self.norm(x_in), self.norm(x_con))
            x_in = x_in + self.mlp(self.norm(x_in)) 

        # Project each token to scalar then reshape back to image size
        x_in = self.fc(x_in)       # [B, 3136, 1]
        x_in = self.relu(x_in)
        x_in = x_in.squeeze(2)     # [B, 3136]
        x_in = x_in.reshape(bs, 1, 56, 56)

        # Upsample to original 224x224 size and add to original image
        x_in = F.interpolate(x_in, size=(224, 224), mode='bilinear', align_corners=False)
        x_in = img + 0.5 * x_in  

        return x_in

    
class Cross_text(nn.Module):
    def __init__(self, dim=768, num_img=32, num_text=3136, depth=2):
        super(Cross_text, self).__init__()
        self.num_img = num_img
        self.num_text = num_text
        self.depth = depth
        self.dim = dim

        # Embed image patches for cross attention conditioning
        self.patch_embed = PatchEmbed(in_c=1, embed_dim=dim)

        # Cross attention module to align text with image features
        self.cross_attn = CrossAttention(dim, dim)

        self.pos_embed_input = nn.Parameter(torch.zeros(1, num_img, dim))
        self.pos_embed_condition = nn.Parameter(torch.zeros(1, num_text, dim))  

        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=3 * dim, act_layer=nn.GELU, drop=0.)
        self.fc = nn.Linear(dim, 1)
        self.relu = nn.ReLU()

        nn.init.trunc_normal_(self.pos_embed_input, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_condition, std=0.02)
        self.apply(_init_weights)
        
    def forward(self, img, text):
        '''
        img: tensor of shape [B, 1, 224, 224] — image
        text: tensor of shape [B, 32, 768] — verb feature tokens
        '''
        bs = img.shape[0]

        x_in = text  # [B, 32, 768]
        x_con = self.patch_embed(img)  # [B, 3136, 768]

        # Add positional encoding
        x_in = x_in + self.pos_embed_input
        x_con = x_con + self.pos_embed_condition

        # Cross-attention update from image to text
        for _ in range(self.depth):
            x_in = self.cross_attn(self.norm(x_in), self.norm(x_con))
            x_in = x_in + self.mlp(self.norm(x_in))

        x_in = text + 0.5 * x_in

        return x_in

        
class self_deq(nn.Module):
    """
    Self-enhancing module with self-attention and MLP refinement.
    This module takes a single feature token [B, 768] and refines it through multiple transformer-style layers.
    """
    def __init__(self, dim = 768, depth = 2):
        super(self_deq, self).__init__()
        self.depth = depth
        self.dim = dim
        
        self.cross_attn = CrossAttention(dim, dim)
        self.norm = nn.LayerNorm(dim)  
        self.mlp = Mlp(in_features=dim, hidden_features=3 *dim, act_layer=nn.GELU, drop=0.)
        self.pos_embed_input = nn.Parameter(torch.zeros(1, 1, dim))
        self.apply(_init_weights)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): input feature of shape [B, 768]
        Returns:
            x (Tensor): enhanced feature of shape [B, 1, 768]
        """
        # Reshape into single token form
        x = x.unsqueeze(1)  # [B, 1, 768]

        # Add learned positional embedding
        x = x + self.pos_embed_input

        # Apply multiple self-attention + MLP blocks
        for i in range(self.depth):
            x = self.cross_attn(self.norm(x), self.norm(x))  # Self-attention
            x = x + self.mlp(self.norm(x))               

        return x
        
def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.trunc_normal_(m, std=0.02)
        
        