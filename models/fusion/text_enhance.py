import torch.nn as nn
from .self_attention import Selftransformer
from .cross_attention import Cross_img, Cross_text
    
class text_image_cross_align(nn.Module):
    def __init__(self, dim=768):
        """
        A module to align image and verb-based textual features using
        both self-attention and cross-attention mechanisms.

        Args:
            dim (int): The embedding dimension used throughout the network.
        """
        super(text_image_cross_align, self).__init__()
        self.dim = dim
        # Self-attention block to enhance verb representation
        self.selfattn_verb = Selftransformer()

        # Cross-attention block: image attends to verb features
        self.cr_img = Cross_img(dim)

        # Cross-attention block: verb attends to image features
        self.cr_text = Cross_text(dim)

    def forward(self, img, verb):
        """
        Forward function for aligning image and verb features.

        Args:
            img (Tensor): [B, 1, 224, 224] - image feature map
            verb (Tensor): [B, 32, 768] - verb-related language embedding

        Returns:
            hot_feature (Tensor): [B, 1, 224, 224] - refined image-like feature map
            traj_feature (Tensor): [B, 32, 768] - refined verb representation
            verb_feat (Tensor): [B, 32, 768] - verb representation passed self-attention blocks
        """
        # First apply self-attention on the verb tokens to enhance intra-textual relations
        verb_feat = self.selfattn_verb(verb, verb, verb)  # [B, 32, 768]

        # Use the enhanced verb features to guide the image refinement
        hot_feature = self.cr_img(img, verb_feat)         # [B, 1, 224, 224]

        # Use the image to guide refinement of the verb features
        traj_feature = self.cr_text(img, verb_feat)       # [B, 32, 768]

        return hot_feature, traj_feature, verb_feat
        