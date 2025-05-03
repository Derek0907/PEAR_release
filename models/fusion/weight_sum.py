import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Sum_weights(nn.Module):
    def __init__(self):
        super(Sum_weights, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  
        self.beta = nn.Parameter(torch.tensor(0.25))  
        self.gamma = nn.Parameter(torch.tensor(0.25)) 
    def forward(self, x1, x2, x3):
        # [b, 224, 224]
        x2_upsampled = F.interpolate(x2.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1) #[b, 224, 224]
        x3_upsampled = F.interpolate(x3.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1) #[b, 224, 224]
        x2_upsampled = x2_upsampled.unsqueeze(1) #[b, 1, 224, 224]
        x3_upsampled = x3_upsampled.unsqueeze(1) #[b, 1, 224, 224]
        x_all = self.alpha * x1 + self.beta * x2_upsampled + self.gamma * x3_upsampled
        return x_all