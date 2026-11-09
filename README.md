# Spatiochromatic Representation in Transformer-Based Discrete Autoencoders

## 1. Objectives

### Research Questions
1. **Global vs. Local Attention:** Standard convolutional discrete autoencoders (VQ-VAEs) (van den Oord, Vinyals and Kavukcuoglu, 2017) are only able to process small, local patches of an image at a time. On the other hand, Vision Transformers (ViTs) (Dosovitskiy et al., 2021) work in a way where every image patch can immediately grasp the entire scene. Therefore, if we swap the convolutional encoder and decoder for Vision Transformers, do human-like basic colour categories still emerge naturally (de Vries et al., 2022; Akbarinia, 2024)?
2. **Spatial vs. Chromatic Tradeoff:** As shown by previous CNN experiments, shrinking the spatial grid negatively affects downstream recognition much faster than shrinking the colour palette (Akbarinia et al., 2020). Does a Transformer's global approach shift this trade-off? Would this let us reach better downstream recognition accuracy at higher compression ratios (Dosovitskiy et al., 2021; Yu et al., 2022)?
3. **Spatiochromatic Mixup in Bigger Codebooks:** In CNN VQ-VAEs, increasing the codebook size causes some vectors to learn spatial patterns and edges instead of individual colours. Taking this into account, does a ViT keep colour and spatial patterns cleaner and more separated, and do its learned colour palettes stay consistent across colour spaces (RGB, CIELAB, DKL) (Akbarinia et al., 2020; Akbarinia and Gil-Rodríguez, 2021)?

### Target & Features
- **Scientific Targets:**
  - Directly compare a ViT-based discrete autoencoder (Yu et al., 2022) against the established CNN VQ-VAE baseline (van den Oord, Vinyals and Kavukcuoglu, 2017; Akbarinia et al., 2020) across matched compression ratios.
  - Analyse the resulting colour palettes across training to see how closely they match human colour categories, as well as how they compare against each other, when trained for **image reconstruction** versus **object classification** (de Vries et al., 2022; Akbarinia, 2024).
- **System & Software Targets:**
  - Create a clean, modular PyTorch codebase where CNN and ViT encoders/decoders can be easily exchanged while keeping the same shared bottlenecks as well as evaluation tools.
  - Set up config files for reproducible experiments, automatic checkpoint saving, and scripts to generate graphs and 3D colour-space scatter plots.
- **Stretch Goals:**
  - Depending on how fast the baseline experiments run and what we find, the setup could allow us to explore different biologically inspired questions that could be of interest, such as: **adaptive token merging** (Bolya et al., 2023) (teaching the model to combine similar pieces of an image to process information faster without losing important details), **colour constancy under lighting shifts** (Akbarinia and Parraga, 2018) (testing if the model recognizes true object colors despite lighting changes), **masked patch prediction** (He et al., 2022) (training the model to fill in missing parts of images from context), or **dense downstream tasks** like object detection and segmentation.