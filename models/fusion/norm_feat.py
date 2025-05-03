import torch.nn as nn
    
class Norm_Feature_free(nn.Module):
    """
    Normalize and align two different feature types into a shared embedding space.
    One input is an image-like feature map, the other is a fused text-image embedding.
    """
    def __init__(self, out_dim):
        super(Norm_Feature_free, self).__init__()
        
        # Project 224x224 image feature to out_dim
        self.fc1 = nn.Linear(224 * 224, out_dim)

        # Project flattened 32x768 token features to out_dim
        self.fc2 = nn.Linear(32 * 768, out_dim)

        self.relu = nn.ReLU()

    def forward(self, feat1, feat2):
        """
        Args:
            feat1 (Tensor): visual feature of shape [B, 1, 224, 224]
            feat2 (Tensor): token sequence feature of shape [B, 32, 768]

        Returns:
            Tuple of two tensors (both of shape [B, out_dim])
        """
        # Flatten and transform image-like feature
        feat1 = feat1.reshape(feat1.shape[0], -1)
        feat1 = self.fc1(feat1)

        # Flatten and transform token-like feature
        feat2 = feat2.reshape(feat2.shape[0], -1)
        feat2 = self.fc2(feat2)

        # Apply ReLU non-linearity
        feat1 = self.relu(feat1)
        feat2 = self.relu(feat2)

        return feat1, feat2
