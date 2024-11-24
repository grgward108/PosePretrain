import torch
import torch.nn as nn
import math


class TemporalTransformer(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, dim_out=3, num_heads=8, depth=5, num_joints=17, maxlen=243, drop_rate=0.1, mask_token_value=0.0):
        super(TemporalTransformer, self).__init__()
        self.dim_in = dim_in
        self.dim_feat = dim_feat
        self.dim_out = dim_out
        self.num_joints = num_joints
        self.maxlen = maxlen
        self.mask_token_value = mask_token_value
        
        # Joint embedding
        self.joint_embed = nn.Linear(dim_in, dim_feat)
        
        # Positional embeddings
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.joint_embed_pos = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        nn.init.trunc_normal_(self.temp_embed, std=0.02)
        nn.init.trunc_normal_(self.joint_embed_pos, std=0.02)
        
        # Transformer layers (spatio-temporal modeling)
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(dim_feat, num_heads=num_heads, mlp_ratio=4.0, drop_rate=drop_rate) for _ in range(depth)
        ])
        
        # Normalization and output layers
        self.norm = nn.LayerNorm(dim_feat)
        self.output_layer = nn.Linear(dim_feat, dim_out)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (B, T, J, C), where C=3.
            mask: Optional binary mask of shape (B, T, J), where 1 indicates masked positions.
                  If None, assumes no masking.
        Returns:
            Predicted 3D joint positions of shape (B, T, J, C).
        """
        B, T, J, C = x.shape
        
        # Embed joints
        x = self.joint_embed(x.view(B * T, J, C))  # (B*T, J, dim_feat)
        x = x.view(B, T, J, -1) + self.temp_embed[:, :T, :, :] + self.joint_embed_pos[:, :J, :]
        
        # Apply masking (replace masked positions with a mask token)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, T, J, 1)
            x = torch.where(mask == 1, torch.full_like(x, self.mask_token_value), x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, T)
        
        # Normalize and project to output
        x = self.norm(x)
        x = self.output_layer(x)
        return x


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim_feat, num_heads, mlp_ratio=4.0, drop_rate=0.1):
        super(TemporalTransformerBlock, self).__init__()
        self.spatial_attn = AttentionBlock(dim_feat, num_heads, drop_rate)
        self.temporal_attn = AttentionBlock(dim_feat, num_heads, drop_rate)
        self.spatial_mlp = MLPBlock(dim_feat, mlp_ratio, drop_rate)
        self.temporal_mlp = MLPBlock(dim_feat, mlp_ratio, drop_rate)

    def forward(self, x, seq_len):
        # Spatial attention and MLP
        x = self.spatial_attn(x) + x
        x = self.spatial_mlp(x) + x

        # Reshape for temporal attention
        B, T, J, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * J, T, C)  # Rearrange for temporal attention

        # Temporal attention and MLP
        x = self.temporal_attn(x, is_temporal=True) + x
        x = self.temporal_mlp(x) + x

        # Reshape back to original shape
        x = x.view(B, J, T, C).permute(0, 2, 1, 3).contiguous()  # Restore original shape
        return x






class AttentionBlock(nn.Module):
    def __init__(self, dim_feat, num_heads, drop_rate):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim_feat, num_heads, dropout=drop_rate, batch_first=True)
        self.norm = nn.LayerNorm(dim_feat)

    def forward(self, x, is_temporal=False):
        if is_temporal:
            # Input shape: (B*J, T, C)
            BJ, T, C = x.shape
            x, _ = self.attention(x, x, x)
            x = self.norm(x)
            return x  # No reshaping needed
        else:
            # Input shape: (B, T, J, C)
            B, T, J, C = x.shape
            x = x.view(B, T * J, C)  # Flatten spatial and temporal dimensions
            x, _ = self.attention(x, x, x)
            x = self.norm(x)
            x = x.view(B, T, J, C)  # Reshape back
            return x




class MLPBlock(nn.Module):
    def __init__(self, dim_feat, mlp_ratio=4.0, drop_rate=0.1):
        super(MLPBlock, self).__init__()
        hidden_dim = int(dim_feat * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim_feat, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, dim_feat),
            nn.Dropout(drop_rate)
        )
        self.norm = nn.LayerNorm(dim_feat)

    def forward(self, x):
        return self.norm(self.mlp(x))


# Utility function for truncating normal initialization
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + torch.erf(x / math.sqrt(2.))) / 2.
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor
