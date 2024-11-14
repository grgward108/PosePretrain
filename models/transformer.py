# transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosePreTrainer(nn.Module):
    def __init__(self, num_markers=143, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, num_parts=10):
        """
        Initialize the PosePreTrainer model.
        
        Args:
            num_markers (int): Number of markers in each frame.
            d_model (int): Embedding dimension for the transformer.
            nhead (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network in the transformer.
            num_parts (int): Number of unique body parts for part-based embeddings.
        """
        super(PosePreTrainer, self).__init__()

        self.num_markers = num_markers
        self.d_model = d_model

        # Input embedding layer for markers
        self.input_proj = nn.Linear(3, d_model)  # Project each marker's (x, y, z) to d_model

        # Part-based embedding
        self.part_embedding = nn.Embedding(num_parts, d_model)
        self.marker_to_part = self._create_marker_to_part_mapping(num_markers, num_parts)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(num_markers, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer for masked marker prediction
        self.output_proj = nn.Linear(d_model, 3)  # Project back to (x, y, z) for each marker

    def _create_marker_to_part_mapping(self, num_markers, num_parts):
        """
        Assign each marker to a part index for part-based embedding.
        
        Args:
            num_markers (int): Total number of markers.
            num_parts (int): Total number of parts.

        Returns:
            torch.Tensor: A tensor of part indices for each marker.
        """
        marker_to_part = torch.randint(0, num_parts, (num_markers,))
        return marker_to_part

    def forward(self, markers, mask=None):
        """
        Forward pass for the PosePreTrainer model.

        Args:
            markers (torch.Tensor): Input marker positions, shape (batch_size, num_markers, 3).
            mask (torch.Tensor): Mask tensor indicating which markers are visible, shape (batch_size, num_markers).

        Returns:
            torch.Tensor: Reconstructed marker positions, shape (batch_size, num_markers, 3).
        """
        batch_size = markers.size(0)

        # Project input markers to embedding space
        marker_embeds = self.input_proj(markers)  # Shape: (batch_size, num_markers, d_model)

        # Add part-based embedding
        part_embeds = self.part_embedding(self.marker_to_part).unsqueeze(0).expand(batch_size, -1, -1)
        marker_embeds = marker_embeds + part_embeds

        # Add positional encoding
        marker_embeds = marker_embeds + self.positional_encoding.unsqueeze(0)

        # Apply masking (if any) to ignore certain markers in the encoder
        if mask is not None:
            marker_embeds = marker_embeds.masked_fill(mask.unsqueeze(-1) == 0, 0)

        # Transformer expects input of shape (num_markers, batch_size, d_model)
        marker_embeds = marker_embeds.permute(1, 0, 2)

        # Transformer Encoder
        encoded_markers = self.transformer_encoder(marker_embeds)  # Shape: (num_markers, batch_size, d_model)

        # Project back to original marker space (x, y, z)
        encoded_markers = encoded_markers.permute(1, 0, 2)  # Shape: (batch_size, num_markers, d_model)
        reconstructed_markers = self.output_proj(encoded_markers)  # Shape: (batch_size, num_markers, 3)

        return reconstructed_markers
