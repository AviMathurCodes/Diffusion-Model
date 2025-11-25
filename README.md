# Diffusion-Model
Denoising Diffusion Probabilistic Model (DDPM) from scratch trained on CIFAR-10

This project is a complete from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on CIFAR-10. The entire diffusion pipeline is written manually in PyTorch, including the forward noising process, the reverse sampling process, sinusoidal timestep embeddings, a custom UNet architecture, and the full training loop.

The goal of this project was to build a modern generative model without relying on high-level diffusion libraries, and to understand every component of the DDPM framework at an implementation level.

## Features

### 1. Full Diffusion Pipeline
- Linear beta schedule over 1000 steps
- Computation of alpha, alpha cumulative products, and noise scales
- Forward process q(x_t | x_0) implemented exactly as defined in the original DDPM paper
- Sanity checks built in to verify that early timesteps resemble the original image and large timesteps look like pure noise

### 2. Timestep Embedding
- Sinusoidal positional encoding similar to Transformer embeddings
- Projection through MLP layers to match the UNet channel dimensions
- Injected into all residual blocks inside the UNet

### 3. Custom UNet Noise Predictor
- Downsampling and upsampling paths with skip connections
- Residual blocks with GroupNorm and GELU/SiLU activations
- Optional spatial attention blocks at lower resolutions
- Bottleneck block with embedded timestep conditioning
- Outputs a noise prediction tensor with shape (batch, 3, 32, 32)

### 4. Training Loop
- Random timestep sampling per batch
- Training objective: MSE between true noise and predicted noise
- Checkpoint saving and loss logging
- Periodic sample generation for monitoring visual fidelity
- Fully GPU-accelerated

### 5. Sampling Procedure
- Implements the ancestral DDPM sampling equation
- Step-by-step denoising from pure Gaussian noise to a 32x32 RGB image
- Supports batch sampling and saving generated images
