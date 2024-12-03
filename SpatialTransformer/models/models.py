
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist  # Add this import

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

    # models.py

    def forward(self, inputs):
        markers, part_labels, mask = inputs

        """
        Forward pass for the SpatialTransformer with masked self-attention.

        Args:
            markers (torch.Tensor): Input markers of shape [batch_size, n_markers, 3].
            part_labels (torch.Tensor): Part labels for each marker [batch_size, n_markers].
            mask (torch.Tensor): Binary mask [batch_size, n_markers], where 1 indicates masked positions.

        Returns:
            torch.Tensor: Reconstructed markers of shape [batch_size, n_markers, 3].
        """

        # After projecting markers
        marker_embeds = self.input_proj(markers)

        # After adding part-based embeddings
        part_embeds = self.part_embedding(part_labels)
        marker_embeds += part_embeds

        # After preparing the mask
        if mask is not None:
            src_key_padding_mask = mask.bool()

        # Before Transformer
        marker_embeds = marker_embeds.permute(1, 0, 2)

        # Transformer Encoder
        transformer_output = self.transformer_encoder(marker_embeds, src_key_padding_mask=src_key_padding_mask)
        transformer_output = transformer_output.permute(1, 0, 2)

        # Before reconstruction
        reconstructed_markers = self.reconstruction_layer(transformer_output)

        return reconstructed_markers


