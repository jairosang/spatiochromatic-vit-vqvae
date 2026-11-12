# Spatiochromatic Representation in Transformer-Based Discrete Autoencoders

**Author:** Jairo San Miguel Sang  
**Supervisor:** Dr. Arash Akbarinia  
**Programme / Institution:** BSc (Hons) Computer Science / Lancaster University  

---

## 1. Objectives

### Research Questions
1. **Global vs. Local Attention:** Standard convolutional discrete autoencoders (VQ-VAEs) (van den Oord, Vinyals and Kavukcuoglu, 2017) are only able to process small, local patches of an image at a time. On the other hand, Vision Transformers (ViTs) (Dosovitskiy et al., 2021) work in a way where every image patch can immediately grasp the entire scene. Therefore, if we swap the convolutional encoder and decoder for Vision Transformers, do human-like basic colour categories still emerge naturally (de Vries et al., 2022; Akbarinia, 2024)?
2. **Spatial vs. Chromatic Tradeoff:** As shown by previous CNN experiments, shrinking the spatial grid negatively affects downstream recognition much faster than shrinking the colour palette (Akbarinia et al., 2020). Does a Transformer's global approach shift this trade-off? Would this let us reach better downstream recognition accuracy at higher compression ratios (Dosovitskiy et al., 2021; Yu et al., 2022)?
3. **Spatiochromatic Mixup in Bigger Codebooks:** In CNN VQ-VAEs, increasing the codebook size causes some vectors to learn spatial patterns and edges instead of individual colours. Taking this into account, does a ViT keep colour and spatial patterns cleaner and more separated, and do its learned colour palettes stay consistent across colour spaces (RGB and YCbCr) (Akbarinia et al., 2020; Akbarinia and Gil-Rodríguez, 2021)?

### Target & Features
- **Scientific Targets:**
  - Directly compare a ViT-based discrete autoencoder (Yu et al., 2022) against the established CNN VQ-VAE baseline (van den Oord, Vinyals and Kavukcuoglu, 2017; Akbarinia et al., 2020) across matched compression ratios.
  - Analyse the resulting colour palettes across training to see how closely they match human colour categories, as well as how they compare against each other, when trained for **image reconstruction** versus **object classification** (de Vries et al., 2022; Akbarinia, 2024).
- **System & Software Targets:**
  - Create a clean, modular PyTorch codebase that easily allows exchanging the baseline CNN checkpoints (Akbarinia et al., 2020) and ViT encoders/decoders while keeping the same shared bottlenecks as well as evaluation tools.
  - Set up config files for reproducible experiments, automatic checkpoint saving, and scripts to generate graphs and 3D colour-space scatter plots.
- **Stretch Goals:**
  - Depending on how fast the baseline experiments run and what we find, the setup could allow us to explore different biologically inspired questions that could be of interest, such as: **adaptive token merging** (Bolya et al., 2023) (teaching the model to combine similar pieces of an image to process information faster without losing important details), **colour constancy under lighting shifts** (Akbarinia and Parraga, 2018) (testing if the model recognizes true object colors despite lighting changes), **masked patch prediction** (He et al., 2022) (training the model to fill in missing parts of images from context), or **dense downstream tasks** like object detection and segmentation.

---

## 2. Related Work

### Published Approaches & State of the Art
- **Discrete Representation Learning:** van den Oord, Vinyals, and Kavukcuoglu (2017) created the VQ-VAE, a neural network that compresses images into a "codebook", or a fixed list of basic visual building blocks. Later, Dosovitskiy et al. (2021) introduced Vision Transformers (ViTs), a new architecture that analyzes images by looking at how different patches of the picture relate to the whole scene. Building on this, Yu et al. (2022) created ViT-VQGAN by swapping older network designs for Transformers. This upgrade resulted in both better-looking image recreations and a deeper understanding of what was in the image.
- **Emergent Colour Categorization in Deep Networks:** Previous research shows that AI models trained to recognize objects naturally learn to group colors much like humans do (de Vries et al., 2022; Akbarinia, 2024). However, experiments on older AI models revealed a surprising contrast: if the model is trained only to recreate the image exactly, it learns clear, human-like colors. But if it is forced to recreate the image and identify objects at the same time, its learned colors turn into muddy, monochromatic textures. Because biological human vision relies heavily on looking at the entire surrounding scene to understand color (Akbarinia and Parraga, 2018), Vision Transformers, designed to look at the big picture, are a perfect tool for studying this issue further.
- **Colour Spaces & Spatiochromatic Compression:** Earlier studies (Akbarinia et al., 2020; Akbarinia and Gil-Rodríguez, 2021), which we will call the CNN baseline, tested older models using various color formats (like RGB and CIELAB) across large sets of images. They discovered that the models always formed similar color groups regardless of the color format or how the model was started. This proves that the natural patterns found in real-world images are what teach the model these colors. Finally, the studies noted that compressing the image data too tightly puts a strict limit on how accurately the original image can be recreated.

### Existing Tools, Libraries & Their Limitations
- Current libraries (timm, vector-quantize-pytorch, HuggingFace diffusers) offer the basic building blocks for Vision Transformers and VQ layers. Meanwhile, the osculari package (Akbarinia, 2023) provides tools based on human vision to test existing models.
- However, current VQ-ViT tools are built to generate highly realistic images using massive codebooks ($K = 1024 \text{--} 8192$), and complex training rules (Yu et al., 2022). Right now, there isn't a ready-to-use system that combines what we actually need: Vision Transformers with very small codebooks ($K = 6\text{--}32$), flexible color-space training, and the classification features required to directly compare these models against the original CNN baseline.

---

## 3. Methods

### Architecture & Algorithms: Baseline CNN vs. Proposed ViT-VQ-VAE
The purpose of this comparison is to isolate a single factor: **whether global self-attention (seeing the whole image at once) or local convolution (seeing only small regions at a time) affects the learned representations.** To ensure a fair comparison, all other components are held constant between the two models; only the encoder and decoder networks are exchanged.

- **Reference CNN baseline:** Will be taken from Akbarinia et al. checkpoints. The model of van den Oord, Vinyals and Kavukcuoglu (2017) and Akbarinia et al. (2020) takes $224 \times 224 \times 3$ images as input and reduces them progressively through stacked strided 2D convolutions and residual blocks, which process only small local regions of the image. The output is a local latent grid of feature vectors, $z_e(x) \in \mathbb{R}^{h \times w \times D}$, at one of four spatial resolutions ($h = w \in \{56, 25, 14, 9\}$). Each vector is then replaced by its nearest neighbour in a learned codebook $E \in \mathbb{R}^{K \times D}$ (using stop-gradient commitment loss); this discrete bottleneck forces the network to compress the image into a limited set of visual building blocks. A symmetric transposed-convolutional decoder reconstructs the $224 \times 224$ image from these codes.
- **Proposed ViT-VQ-VAE:** This model takes the same $224 \times 224 \times 3$ input as the baseline but divides it into non-overlapping square patches of a variable size $P$. Each patch becomes a token, augmented with 2D positional embeddings to retain spatial layout. These pass through Multi-Head Self-Attention Transformer blocks, allowing every patch to attend to the entire scene; a global processing approach that fundamentally contrasts with the CNN's local receptive fields. The tokens then pass through the discrete codebook $E$, and a symmetric ViT decoder reconstructs the full $224 \times 224$ image from the quantized tokens.
  - *Execution:*
       1. *Hyperparameter Optimization:* We will conduct hyperparameter optimization on patch sizes $P \in \{4, 8, 16, 32\}$ (corresponding to grid resolutions $56\times56$, $28\times28$, $14\times14$, and $7\times7$) on a validation subset (ImageNet-100) to find the optimal configuration balancing compute, reconstruction quality, and token fidelity
       2. *Full Evaluation:* Utilize the optimal patch size $P*$ to the primary downstream training, object recognition and cross-colour-space evaluations.

#### Optimization and training regimes:
- Unlike CNNs, Transformers do not possess built-in translation equivariance and therefore require a more careful optimization setup. All models will be trained from scratch using the **AdamW** optimizer with linear learning-rate warmup and cosine decay. If necessary, EMA and $\ell_2$-normalized codebook updates (Yu et al., 2022) will be applied to prevent unused (dead) codebook entries.
- All ViT models will be trained from scratch under two regimes:
  1. **Pure Reconstruction:** the objective is faithful image reconstruction, measured by pixel MSE plus the VQ commitment loss, including symmetric and cross-colour-space pairs restricted to RGB and YCbCr (Akbarinia and Gil-Rodríguez, 2021).
  2. **Joint Object Recognition:** a classification head is attached directly to the quantized bottleneck, so the model must both compress the image and support object recognition.

### Datasets & Experimental Setup
- **Datasets:** We will use ImageNet-1K (Russakovsky et al., 2015) as well as smaller datasets like ImageNet-100 and MS-COCO, keeping image sizes at $224 \times 224$. To be efficient, we will extensively test our different model settings on the smaller datasets first, and then apply the most successful setups to the full ImageNet-1K dataset.
- **Model Settings (Hyperparameters):**
  - **Codebook sizes ($K$):** To directly compare with the previous CNN study, we will use sizes of $$K \in \{6, 8, 11, 16, 32\}$$. We will also test a much larger size ($K = 1024$) on low-resolution images to see how the model handles extreme compression rates.
  - **Reliability:** To ensure our results are consistent and not just due to chance, we will run each configuration three separate times using different random seeds ($N = 3$).
- **Development Process & Testing:**
  - **Code Organization:** The project will be managed in a structured GitHub repository. We will separate the code into clear categories for the core models (`backbones/` for CNN vs. ViT), `quantizers/`, `colour_spaces/`, and `analysis/`.
  - **Automated Testing:** We will use `pytest` to automatically check our code for errors. These tests will verify that our color-space math works in reverse, our compression ratios are exact, our training signals (gradients) flow properly, and our data shapes remain correct throughout the process.

---

## 4. Plan for Analysis

### Baselines & Benchmarks
- Main Comparison: We will directly compare our model to the standard CNN VQ-VAE baseline (Akbarinia et al., 2020). Furthermore, our ViT will match the CNN baseline in number of parameters, compression levels, codebook sizes ($K$), and training time (epochs); and will be compared against the frozen unchanged baseline checkpoints.
- Measuring Compression: We will calculate how well the model compresses the data using the bit-level compression ratio formula:

  $$\text{Bit Compression Ratio} = \frac{H \times W \times C \times \text{BitsPerPixel}}{h \times w \times \log_2(K)}$$

### Evaluation Metrics & Statistical Analysis
To ensure direct comparability, I will replicate all four core evaluation methods from the CNN baseline and extend them with Transformer-specific diagnostics:
1. **Downstream Recognition & Reconstruction Frontier:**
   - Replicate the **ViT-B/16 Object Recognition Accuracy vs. Bit Compression Ratio** benchmark by passing CNN- and ViT-reconstructed validation images into a frozen pretrained ViT-B/16 evaluator (Dosovitskiy et al., 2021).
   - Complement this with pixel fidelity (PSNR, SSIM) and perceptual colour difference ($\Delta E_{00}$ / CIEDE2000) (Sharma, Wu and Dalal, 2005).
2. **Chromatic & Spatial Representation Analysis:**
   - **Single- & Multi-Vector Uniform Decoding (Swatch Analysis):** Following the CNN baseline protocol (Akbarinia et al., 2020), populate the entire $h \times w$ bottleneck grid with a single embedding vector $e_i$ (as well as spatial pairs $e_i, e_j$) and decode to inspect whether ViT tokens learn flat focal colours or spatial edge/stripe patterns (e.g., at $K = 32$), while checking whether learned 2D positional embeddings introduce spatial border effects.
   - **3D Cross-Instance Colour Clustering:** Project the decoded RGB/YCbCr pixel distributions across all $N = 3$ network instances into 3D colour space and quantify inter-seed cluster consistency (centroid variance after Hungarian matching) and alignment with human basic colour categories (de Vries et al., 2022; Akbarinia, 2024).
   - **Codebook Utilization:** Track codebook perplexity and active-code percentage across $K$ so CNN vs. ViT differences aren't confounded by dead vectors (Yu et al., 2022).
   - **Self-Attention Context Probing (ViT-Specific):** Extract pre-quantization self-attention maps (Dosovitskiy et al., 2021) to measure whether tokens attend locally (mimicking CNNs) or pool long-range scene context before selecting a discrete colour vector.

---

## 5. Resources

### Compute & Hardware
- **GPUs:** Lancaster University Leipzig computing resources (to be specified later on).
- **Storage:** ~150 GBs (covering dataset subsets, model checkpoints across random seeds, and generated evaluation outputs).

**Computational Scope & Efficiency:**
Compared to the original design, the revised methodology significantly reduces the computational burden:

- *Scope Reductions*: A significant portion of compute savings are achieved by reducing the random seeds from $N=5$ to $N=3$ (a 40% reduction per configuration), reducing the evaluated codebook sizes from 8 to 5 (a 37.5% reduction), and eliminating CNN retraining entirely by using existing checkpoints (a ~50% reduction in total model training).
- *Avoiding Grid Search*: Restricting patch-size selection via an initial Hyperparameter Optimization (HPO) stage prevents an exponential combinatorial explosion. This reduces the experimental grid from 160 full training runs (4 patch sizes × 8 codebooks × 5 seeds) down to approximately 25 total runs.
- *Colour Transformations*: Restricting cross-colour-space targets strictly to RGB and YCbCr eliminates the computationally heavy non-linear mathematical overhead (e.g., cubic roots) required by perceptual spaces like CIELAB during the loss calculation, maximizing GPU throughput.

### Software, APIs & Licences
- **Core Stack:** `Python 3.10+`, `PyTorch`, `timm` (Vision Transformer components), `scikit-image` (colour space transformations and metrics), `TensorBoard` (experiment tracking).
- **Data Access:** Public academic research access to ImageNet (Russakovsky et al., 2015).

---

## 6. Workplan

![image info](./attachments/images/project_timeline.png)

---

## 7. References

- **Akbarinia, A.** (2023) *Osculari: A Python package to explore and interpret deep neural networks (v0.0.3)*. Zenodo. Available at: https://doi.org/10.5281/zenodo.10268104.
- **Akbarinia, A.** (2024) 'Exploring the categorical nature of colour perception: Insights from artificial networks', *Neural Networks*, 181, p. 106758. Available at: https://doi.org/10.1016/j.neunet.2024.106758.
- **Akbarinia, A. and Gil-Rodríguez, R.** (2021) 'Color Conversion in Deep Autoencoders', *Journal of Perceptual Imaging*, 4(1), p. 010501. Available at: https://doi.org/10.2352/J.Percept.Imaging.2021.4.1.010501.
- **Akbarinia, A., Gil-Rodríguez, R., Flachot, A. and Toscani, M.** (2020) 'The Utility of Decorrelating Colour Spaces in Vector Quantised Variational Autoencoders', *arXiv preprint arXiv:2009.14487*. Available at: https://arxiv.org/abs/2009.14487.
- **Akbarinia, A. and Parraga, C.A.** (2018) 'Colour Constancy Beyond the Classical Receptive Field', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(9), pp. 2081–2094. Available at: https://doi.org/10.1109/TPAMI.2017.2753239.
- **Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C. and Hoffman, J.** (2023) 'Token Merging: Your ViT But Faster', *International Conference on Learning Representations (ICLR)*. Available at: https://arxiv.org/abs/2210.09461.
- **de Vries, J.P., Akbarinia, A., Flachot, A. and Gegenfurtner, K.R.** (2022) 'Emergent color categorization in a neural network trained for object recognition', *eLife*, 11, p. e76472. Available at: https://doi.org/10.7554/eLife.76472.
- **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N.** (2021) 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', *International Conference on Learning Representations (ICLR)*. Available at: https://arxiv.org/abs/2010.11929.
- **He, K., Chen, X., Xie, S., Li, Y., Dollár, P. and Girshick, R.** (2022) 'Masked Autoencoders Are Scalable Vision Learners', *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16000–16009. Available at: https://arxiv.org/abs/2111.06377.
- **Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C. and Fei-Fei, L.** (2015) 'ImageNet Large Scale Visual Recognition Challenge', *International Journal of Computer Vision*, 115(3), pp. 211–252. Available at: https://doi.org/10.1007/s11263-015-0816-y.
- **Sharma, G., Wu, W. and Dalal, E.N.** (2005) 'The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations', *Color Research & Application*, 30(1), pp. 21–30. Available at: https://doi.org/10.1002/col.20070.
- **van den Oord, A., Vinyals, O. and Kavukcuoglu, K.** (2017) 'Neural Discrete Representation Learning', *Advances in Neural Information Processing Systems (NeurIPS)*, 30. Available at: https://arxiv.org/abs/1711.00937.
- **Yu, J., Li, X., Koh, J.Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J. and Wu, Y.** (2022) 'Vector-Quantized Image Modeling with Improved VQGAN', *International Conference on Learning Representations (ICLR)*. Available at: https://arxiv.org/abs/2110.04627.