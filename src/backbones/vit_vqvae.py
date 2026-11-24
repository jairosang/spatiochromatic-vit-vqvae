"""
ViT-VQ-VAE: a discrete autoencoder that uses a Vision Transformer encoder
and decoder.

    image (B, 3, 224, 224)
      -> ViTEncoder   split into P x P patches, then self-attention over all patches
      -> VectorQuantizer  replace each token with its nearest of K codebook vectors
      -> ViTDecoder   self-attention over the quantised tokens, then back to pixels
    reconstruction (B, 3, 224, 224)

With patch size P the latent grid is (224 / P) x (224 / P):
    P = 4  -> 56 x 56,   P = 8  -> 28 x 28,
    P = 16 -> 14 x 14,   P = 32 ->  7 x  7.

An optional classification head on the quantised tokens supports the
"joint object recognition" training regime.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange

from src.quantizers.vector_quantizer import VectorQuantizer


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block: x + Attention(x), then x + MLP(x)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attention_out, _ = self.attention(h, h, h, need_weights=False)
        x = x + attention_out
        x = x + self.mlp(self.norm2(x))
        return x


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, num_patches, P*P*C), patches in row-major order."""
    return rearrange(
        images, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
    )


def unpatchify(patches: torch.Tensor, patch_size: int, grid_size: int) -> torch.Tensor:
    """(B, num_patches, P*P*C) -> (B, C, H, W), the inverse of patchify."""
    return rearrange(
        patches,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=grid_size, w=grid_size, p1=patch_size, p2=patch_size,
    )


# ---------------------------------------------------------------------------
# Encoder and decoder
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """Image -> one latent vector per patch, shape (B, num_patches, code_dim)."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float, code_dim: int, output_scale: float = 1.0):
        """output_scale: initial RMS of the latents; set it to the codebook's scale."""
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels

        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        # One learned embedding per grid position (row, col) to keep the spatial layout.
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Project down to the (usually smaller) codebook dimension.
        self.to_code_dim = nn.Linear(embed_dim, code_dim)
        # Keep the latents at the codebook's scale. Without this, the encoder output
        # grows freely during training while the codebook (moved by only ~lr per step)
        # cannot follow, so the VQ loss keeps rising. Same role as the final
        # BatchNorm (weight = 1/40) at the end of the CNN baseline's encoder.
        self.code_norm = nn.LayerNorm(code_dim)
        nn.init.constant_(self.code_norm.weight, output_scale)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = patchify(images, self.patch_size)        # (B, N, P*P*C)
        x = self.patch_embedding(x)                  # (B, N, embed_dim)
        x = x + self.position_embedding
        for block in self.blocks:                    # every patch attends to every patch
            x = block(x)
        x = self.norm(x)
        return self.code_norm(self.to_code_dim(x))   # (B, N, code_dim)


class ViTDecoder(nn.Module):
    """Quantised tokens (B, num_patches, code_dim) -> image (B, C, H, W)."""

    def __init__(self, image_size: int, patch_size: int, out_channels: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float, code_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        num_patches = self.grid_size ** 2
        patch_dim = patch_size * patch_size * out_channels

        self.from_code_dim = nn.Linear(code_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.to_pixels = nn.Linear(embed_dim, patch_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.from_code_dim(tokens)               # (B, N, embed_dim)
        x = x + self.position_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.to_pixels(x)                        # (B, N, P*P*C)
        return unpatchify(x, self.patch_size, self.grid_size)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ViTVQVAE(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256, encoder_depth: int = 4, decoder_depth: int = 4, num_heads: int = 4, mlp_ratio: float = 4.0, num_codes: int = 8, code_dim: int = 64, commitment_beta: float = 0.25, l2_normalize_codes: bool = False, dead_code_threshold: float = 1.0, num_classes: int | None = None):
        """
        Args:
            image_size: input height and width (images must be square).
            patch_size: P, one of {4, 8, 16, 32} in the proposal.
            in_channels: 3 for RGB or YCbCr.
            embed_dim / *_depth / num_heads / mlp_ratio: Transformer size.
            num_codes: K, the codebook size (6, 8, 11, 16, 32, ...).
            code_dim: D, the size of each codebook vector.
            commitment_beta: VQ commitment loss weight.
            l2_normalize_codes: cosine-similarity codebook lookup (Yu et al., 2022).
            dead_code_threshold: restart codes used less than this many times per
                batch (running average); 0 turns restarts off.
            num_classes: if set, add a classification head on the quantised
                tokens for the joint reconstruction + recognition regime.
        """
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = image_size // patch_size
        self.num_codes = num_codes

        self.quantizer = VectorQuantizer(
            num_codes, code_dim, commitment_beta, l2_normalize_codes, dead_code_threshold
        )
        # Start the encoder output at the same scale as the codebook entries.
        codebook_scale = self.quantizer.codebook.weight.detach().std().item()
        self.encoder = ViTEncoder(
            image_size, patch_size, in_channels,
            embed_dim, encoder_depth, num_heads, mlp_ratio, code_dim,
            output_scale=codebook_scale,
        )
        self.decoder = ViTDecoder(
            image_size, patch_size, in_channels,
            embed_dim, decoder_depth, num_heads, mlp_ratio, code_dim,
        )

        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(code_dim),
                nn.Linear(code_dim, num_classes),
            )

    # -- main pass ----------------------------------------------------------

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images: (B, C, H, W)

        Returns a dict with:
            reconstruction: (B, C, H, W)
            indices:        (B, h, w)  chosen code for each grid cell
            vq_loss:        scalar
            perplexity:     scalar
            active_codes:   scalar
            logits:         (B, num_classes), or None when there is no classifier
        """
        z = self.encoder(images)                     # (B, N, D)
        vq = self.quantizer(z)
        quantized = vq["quantized"]                  # (B, N, D)
        reconstruction = self.decoder(quantized)

        logits = None
        if self.classifier is not None:
            # Average the quantised tokens over the grid, then classify.
            logits = self.classifier(quantized.mean(dim=1))

        return {
            "reconstruction": reconstruction,
            "indices": vq["indices"].view(-1, self.grid_size, self.grid_size),
            "vq_loss": vq["vq_loss"],
            "perplexity": vq["perplexity"],
            "active_codes": vq["active_codes"],
            "logits": logits,
        }

    # -- helpers for analysis -----------------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, images: torch.Tensor) -> torch.Tensor:
        """Images (B, C, H, W) -> code indices (B, h, w)."""
        z = self.encoder(images)
        indices = self.quantizer(z)["indices"]
        return indices.view(-1, self.grid_size, self.grid_size)

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Code indices (B, h, w) -> images (B, C, H, W).

        Used for swatch analysis, e.g. fill the whole grid with code i:
            grid = torch.full((1, model.grid_size, model.grid_size), i)
            swatch = model.decode_from_indices(grid)
        """
        tokens = self.quantizer.lookup(indices.flatten(1))   # (B, N, D)
        return self.decoder(tokens)

    def bit_compression_ratio(self, bits_per_pixel: int = 8) -> float:
        """(H * W * C * bits_per_pixel) / (h * w * log2(K)), as in the proposal."""
        original_bits = self.image_size ** 2 * self.in_channels * bits_per_pixel
        latent_bits = self.grid_size ** 2 * math.log2(self.num_codes)
        return original_bits / latent_bits
