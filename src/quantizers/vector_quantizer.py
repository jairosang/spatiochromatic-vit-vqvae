"""
Vector quantiser (the discrete bottleneck) from van den Oord et al. (2017).

It is kept separate from the encoder/decoder so the same bottleneck can be
shared between the CNN baseline and the ViT model.

Each input vector is replaced by its nearest neighbour in a learned codebook
E of shape (K, D). Gradients skip the non-differentiable lookup through the
straight-through estimator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, commitment_beta: float = 0.25, l2_normalize: bool = False,):
        """
        Args:
            num_codes: K, the number of codebook vectors (e.g. 6, 8, 11, 16, 32).
            code_dim: D, the size of each codebook vector.
            commitment_beta: weight of the commitment loss (0.25 in the VQ-VAE paper).
            l2_normalize: if True, compare vectors by cosine similarity instead of
                Euclidean distance (Yu et al., 2022). This helps prevent dead codes.
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_beta = commitment_beta
        self.l2_normalize = l2_normalize

        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def get_codebook(self) -> torch.Tensor:
        """The (K, D) codebook, normalised if l2_normalize is on."""
        codebook = self.codebook.weight
        if self.l2_normalize:
            codebook = F.normalize(codebook, dim=-1)
        return codebook

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Turn integer code indices (any shape) into codebook vectors (..., D)."""
        return self.get_codebook()[indices]

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: encoder output of shape (B, N, D), with N tokens per image.

        Returns a dict with:
            quantized:    (B, N, D) quantised vectors (straight-through gradients)
            indices:      (B, N)    index of the chosen code for each token
            vq_loss:      scalar    codebook loss + beta * commitment loss
            perplexity:   scalar    effective number of codes in use (max K)
            active_codes: scalar    fraction of codes used in this batch
        """
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        codebook = self.get_codebook()

        # 1. Find the nearest codebook vector for every token.
        #    ||z - e||^2 = ||z||^2 - 2 z.e + ||e||^2
        flat_z = z.reshape(-1, self.code_dim)  # (B*N, D)
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )  # (B*N, K)
        indices = distances.argmin(dim=1)  # (B*N,)
        quantized = codebook[indices].view_as(z)  # (B, N, D)

        # 2. VQ losses (with stop-gradient via .detach()).
        #    codebook loss:   moves the codes towards the encoder outputs
        #    commitment loss: keeps the encoder outputs close to their chosen code
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss

        # 3. Straight-through estimator: the forward pass uses the quantised
        #    value, while the backward pass copies gradients straight to z.
        quantized = z + (quantized - z).detach()

        # 4. Codebook usage statistics (used to spot dead codes).
        with torch.no_grad():
            code_probs = F.one_hot(indices, self.num_codes).float().mean(dim=0)
            perplexity = torch.exp(-(code_probs * torch.log(code_probs + 1e-10)).sum())
            active_codes = (code_probs > 0).float().mean()

        return {
            "quantized": quantized,
            "indices": indices.view(z.shape[:-1]),
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "active_codes": active_codes,
        }
