import torch
import torch.nn as nn

class SpatialTransformer(nn.Module):
    def __init__(self, n_markers=143, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1, n_parts=9):
        """
        Transformer-based model for reconstructing masked body markers with part-based embeddings.

        Args:
            n_markers (int): Number of body markers.
            embed_dim (int): Dimensionality of the embeddings.
            num_heads (int): Number of attention heads in each layer.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate.
            n_parts (int): Number of body parts for part-based embeddings.
        """
        super(SpatialTransformer, self).__init__()
        self.n_markers = n_markers
        self.embed_dim = embed_dim

        # Marker feature projection (3D -> embed_dim)
        self.input_proj = nn.Linear(3, embed_dim)

        # Learnable part-based embeddings
        self.part_embedding = nn.Embedding(n_parts, embed_dim)

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu'
            ),
            num_layers=num_layers
        )

        # Output reconstruction layer (embed_dim -> 3D)
        self.reconstruction_layer = nn.Linear(embed_dim, 3)

    def forward(self, markers, part_labels, mask=None):
        """
        Forward pass for the SpatialTransformer.

        Args:
            markers (torch.Tensor): Input markers of shape [batch_size, n_markers, 3].
            part_labels (torch.Tensor): Part labels for each marker [batch_size, n_markers].
            mask (torch.Tensor): Optional binary mask [batch_size, n_markers].

        Returns:
            torch.Tensor: Reconstructed markers of shape [batch_size, n_markers, 3].
        """
        batch_size = markers.size(0)

        # Project marker features to embedding dimension
        marker_embeds = self.input_proj(markers)  # Shape: (bs, n_markers, embed_dim)

        # Apply learnable part-based embeddings
        part_embeds = self.part_embedding(part_labels)  # Shape: (bs, n_markers, embed_dim)
        marker_embeds = marker_embeds + part_embeds

        # Transformer Encoder
        marker_embeds = marker_embeds.permute(1, 0, 2)  # Shape: (n_markers, bs, embed_dim)
        transformer_output = self.transformer_encoder(marker_embeds)
        transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (bs, n_markers, embed_dim)

        # Reconstruct marker positions
        reconstructed_markers = self.reconstruction_layer(transformer_output)  # Shape: (bs, n_markers, 3)

        return reconstructed_markers
