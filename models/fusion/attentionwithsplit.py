import torch.nn as nn
from .cross_attention import self_deq

class Mlp(nn.Module):
    """
    Multi-layer perceptron block used in transformer-based architectures.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of MLP.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    
    
class Att_split(nn.Module):
    """
    Attention-based feature splitter.
    This module splits a shared latent feature into three branches (pose, area, manipulation).
    """
    def __init__(self, input_dim, output_dim1, output_dim2):
        super(Att_split, self).__init__()
        self.input_dim = input_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        
        # Transformer-like feature enhancement
        self.transformer = self_deq()
        
        # Linear projection from [B, input_dim, 1] → [B, input_dim, 3]
        self.fc = nn.Linear(1, 3)
        self.relu = nn.ReLU()
        
        # Optional further MLP refinement for each branch (currently unused)
        self.enhance1 = Mlp(768, 768 * 3, 768)
        self.enhance2 = Mlp(768, 768 * 3, 768)
        self.enhance3 = Mlp(768, 768 * 3, 768)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature of shape [B, input_dim]
        Returns:
            x1, x2, x3 (Tensor): Three feature splits of shape [B, input_dim], 
                                 corresponding to pose, area, and manipulation tasks.
        """
        x = self.transformer(x)  # [B, 1, input_dim]
        x = x.permute(0, 2, 1)   # [B, input_dim, 1]
        x = self.fc(x)           # [B, input_dim, 3]
        x = self.relu(x)

        x1 = x[:, :, 0]  # Pose feature branch
        x2 = x[:, :, 1]  # Area feature branch
        x3 = x[:, :, 2]  # Manipulation feature branch

        return x1, x2, x3
        

