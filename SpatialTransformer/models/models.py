
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
        print(f"[Rank {dist.get_rank()}] Inside model.forward:")
        print(f"  markers.device: {markers.device}")
        print(f"  part_labels.device: {part_labels.device}")
        print(f"  mask.device: {mask.device}")
        print(f"  self.input_proj.weight.device: {self.input_proj.weight.device}")

        # After projecting markers
        marker_embeds = self.input_proj(markers)
        print(f"[Rank {dist.get_rank()}] marker_embeds.device after input_proj: {marker_embeds.device}")

        # After adding part-based embeddings
        part_embeds = self.part_embedding(part_labels)
        marker_embeds += part_embeds
        print(f"[Rank {dist.get_rank()}] marker_embeds.device after adding part_embeds: {marker_embeds.device}")

        # After preparing the mask
        if mask is not None:
            src_key_padding_mask = mask.bool()
            print(f"[Rank {dist.get_rank()}] src_key_padding_mask.device: {src_key_padding_mask.device}")

        # Before Transformer
        marker_embeds = marker_embeds.permute(1, 0, 2)
        print(f"[Rank {dist.get_rank()}] marker_embeds.device before Transformer: {marker_embeds.device}")

        # Transformer Encoder
        transformer_output = self.transformer_encoder(marker_embeds, src_key_padding_mask=src_key_padding_mask)
        transformer_output = transformer_output.permute(1, 0, 2)
        print(f"[Rank {dist.get_rank()}] transformer_output.device: {transformer_output.device}")

        # Before reconstruction
        reconstructed_markers = self.reconstruction_layer(transformer_output)
        print(f"[Rank {dist.get_rank()}] reconstructed_markers.device: {reconstructed_markers.device}")


