# Spatiochromatic Representation in Transformer-Based Discrete Autoencoders

BSc (Hons) Computer Science final-year project, Lancaster University.
**Author:** Jairo San Miguel Sang · **Supervisor:** Dr. Arash Akbarinia

This project investigates what happens when the convolutional encoder and decoder of a Vector-Quantised Variational Autoencoder (VQ-VAE) are replaced with Vision Transformers. Convolutional VQ-VAEs trained on natural images are known to learn small colour palettes that resemble human basic colour categories. Because a Transformer can attend to the whole scene at once rather than to small local neighbourhoods, it may change how colour and spatial information are traded off in the discrete code. We compare a ViT-based discrete autoencoder against an established CNN baseline under matched compression.

## Research questions

1. **Global vs. local attention:** do human-like colour categories still emerge when the CNN is swapped for a ViT?
2. **Spatial vs. chromatic trade-off:** does global attention shift the balance between spatial resolution and palette size, and so improve downstream recognition at high compression?
3. **Spatiochromatic mixup:** with larger codebooks, does a ViT keep colour and spatial patterns more cleanly separated than a CNN, and are its palettes consistent across colour spaces (RGB, YCbCr)?

The full motivation, related work, methodology and bibliography are in [`papers/proposal.md`](papers/proposal.md).

## Approach

```
image → ViT encoder → discrete codebook (K codes) → ViT decoder → reconstruction
                                  └─ optional classification head
```

Experiments vary the factors that determine the compression ratio and what the model is encouraged to learn:

- **Spatial grid:** patch size, which sets the resolution of the latent grid.
- **Palette size:** number of codebook entries, kept small (roughly 6–32) so individual codes can be inspected as colours.
- **Colour space:** the space of the input and of the reconstruction target.
- **Training objective:** reconstruction only, or reconstruction jointly with object classification.

The resulting codebooks are analysed against human colour categories and compared with the CNN baseline.

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | Model, quantiser, colour-space and data-loading code |
| `papers/` | Project proposal and thesis write-up |
| `prototyping/` | Exploratory notebooks |
| `data/` | Local datasets (git-ignored) |

## Getting started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Datasets are expected under `data/`. Imagenette is used for quick local prototyping; full experiments run on ImageNet.

## Key references

- van den Oord, Vinyals and Kavukcuoglu (2017). *Neural Discrete Representation Learning.* (VQ-VAE)
- Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Yu et al. (2022). *Vector-quantized Image Modeling with Improved VQGAN.* (ViT-VQGAN)
- Akbarinia, Gil-Rodríguez, Flachot and Toscani (2020). *The Utility of Decorrelating Colour Spaces in Vector Quantised Variational Autoencoders.* [arXiv:2009.14487](https://arxiv.org/abs/2009.14487) (CNN baseline)
- de Vries, Akbarinia, Flachot and Gegenfurtner (2022). *Emergent color categorization in a neural network trained for object recognition.* [eLife 11:e76472](https://doi.org/10.7554/eLife.76472)

See the proposal for the complete bibliography.
