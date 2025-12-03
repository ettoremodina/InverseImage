# Growing Neural Cellular Automata - A PyTorch Implementation

## Overview

This project implements the **Growing Neural Cellular Automata** model from the Distill publication by Mordvintsev et al. (2020). The model demonstrates how simple local rules, learned through gradient descent, can produce complex global behaviors such as pattern growth, persistence, and regeneration.

The core idea is inspired by biological morphogenesis: how a single fertilized egg cell can reliably self-assemble into complex multicellular organisms through local cell-to-cell communication.

---

## Table of Contents

1. [Biological Motivation](#1-biological-motivation)
2. [Cellular Automata Background](#2-cellular-automata-background)
3. [Model Architecture](#3-model-architecture)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Implementation Details](#5-implementation-details)
6. [Training Experiments](#6-training-experiments)
7. [Code Reference](#7-code-reference)

---

## 1. Biological Motivation

Living organisms possess remarkable capabilities:
- **Morphogenesis**: Growing from a single cell into complex anatomies
- **Self-organization**: Cells communicate locally to decide global body plans
- **Regeneration**: Some species (like salamanders) can repair damaged organs

The goal is to create computational models that capture these properties - systems that can:
- Grow patterns from a single seed cell
- Maintain stable patterns over time
- Regenerate from damage

---

## 2. Cellular Automata Background

**Classical Cellular Automata (CA)**:
- Grid of cells updated iteratively
- Same rules applied to each cell
- New state depends only on local neighborhood (typically 3×3)
- Famous examples: Conway's Game of Life, Rule 110

**Neural Cellular Automata (NCA)**:
- Use continuous values instead of discrete states
- Update rule is a differentiable neural network
- Can be trained via backpropagation
- Same network weights shared across all cells (local, uniform rules)

---

## 3. Model Architecture

### 3.1 Cell State Representation

Each cell is represented as a **16-dimensional vector**:

| Channels | Description |
|----------|-------------|
| 0-2 (RGB) | Visible color channels, values in [0, 1] |
| 3 (Alpha) | "Alive" indicator - cells with α > 0.1 are mature |
| 4-15 (Hidden) | Hidden state for internal communication |

```
Cell State: [R, G, B, α, h₁, h₂, ..., h₁₂] ∈ ℝ¹⁶
```

The hidden channels act as chemical signals or internal memory that cells use to coordinate behavior.

### 3.2 Update Rule Pipeline

A single update step consists of four sequential phases:

```
Input State → Perception → Update Rule → Stochastic Update → Alive Masking → Output State
```

#### Phase 1: Perception

Cells perceive their local environment using **Sobel filters** to estimate spatial gradients:

- **Identity filter**: Current cell state
- **Sobel-X**: Horizontal gradient (∂/∂x)
- **Sobel-Y**: Vertical gradient (∂/∂y)

This produces a 48-dimensional perception vector per cell (16 channels × 3 filters).

#### Phase 2: Update Rule (The "Brain")

A small neural network processes the perception vector:

```
perception (48) → Dense(128) → ReLU → Dense(16) → state_update
```

Implemented as 1×1 convolutions for efficiency:
- ~8,000 trainable parameters
- Final layer initialized to zero (do-nothing initially)
- No ReLU on output (updates can be positive or negative)

#### Phase 3: Stochastic Cell Update

Cells update asynchronously (no global clock):
- Each cell has 50% probability of updating per step
- Models the fact that real cells operate independently
- Implemented as per-cell dropout on update vectors

#### Phase 4: Alive Masking

Maintains the growth boundary:
- A cell is "alive" if it or any neighbor has α > 0.1
- Dead cells have all channels set to 0
- Prevents computation outside the organism

---

## 4. Mathematical Formulation

### 4.1 Perception (Sobel Filters)

The Sobel operators estimate image gradients:

$$
\text{Sobel}_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad
\text{Sobel}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}
$$

For each cell position $(i,j)$ and channel $c$:

$$
\nabla_x S_{c}^{(i,j)} = \text{Sobel}_x * S_c, \quad \nabla_y S_{c}^{(i,j)} = \text{Sobel}_y * S_c
$$

The perception vector concatenates:

$$
P^{(i,j)} = [S^{(i,j)}, \nabla_x S^{(i,j)}, \nabla_y S^{(i,j)}] \in \mathbb{R}^{48}
$$

### 4.2 Update Rule

The neural network computes state deltas:

$$
\Delta S = W_2 \cdot \text{ReLU}(W_1 \cdot P + b_1)
$$

where:
- $W_1 \in \mathbb{R}^{128 \times 48}$, $b_1 \in \mathbb{R}^{128}$
- $W_2 \in \mathbb{R}^{16 \times 128}$ (initialized to zero)

### 4.3 Stochastic Update

$$
S_{t+1} = S_t + \Delta S \odot M
$$

where $M \sim \text{Bernoulli}(p=0.5)$ is a per-cell binary mask.

### 4.4 Alive Masking

$$
\text{alive}(S) = \text{MaxPool}_{3\times3}(S_{[\alpha]}) > 0.1
$$

$$
S_{t+1} = S_{t+1} \odot \text{alive}(S_{t+1})
$$

### 4.5 Loss Function

Pixel-wise L2 loss between RGBA channels and target:

$$
\mathcal{L} = \sum_{i,j} \| S_{[:4]}^{(i,j)} - T^{(i,j)} \|_2^2
$$

---

## 5. Implementation Details

### 5.1 Hyperparameters

From `core.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHANNEL_N` | 16 | Total channels per cell |
| `TARGET_SIZE` | 40 | Grid dimensions (40×40) |
| `POOL_SIZE` | 1024 | Sample pool size |
| `CELL_FIRE_RATE` | 0.5 | Update probability |

### 5.2 Perception Filters

```python
# From core.py
filters = torch.stack([
    torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]),      # Identity
    torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]),   # Sobel-X
    torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]).T  # Sobel-Y
])
```

### 5.3 Per-Channel Convolution

Applies filters to each channel independently:

```python
def perchannel_conv(x, filters):
    b, c, h, w = x.shape
    y = x.reshape(b * c, 1, h, w)
    y = F.pad(y, (1, 1, 1, 1), mode='circular')  # Circular padding
    y = F.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)  # Output: [B, C*3, H, W]
```

### 5.4 Alive Detection

```python
def alive(x, threshold=0.1):
    x = F.pad(x, (1, 1, 1, 1), mode='circular')
    return F.max_pool2d(x, 3, stride=1, padding=0) > threshold
```

### 5.5 CA Model Architecture

```python
class CAModel(nn.Module):
    def __init__(self, channel_n, update_rate=0.5):
        super().__init__()
        self.channel_n = channel_n
        self.update_rate = update_rate
        
        self.brain = nn.Sequential(
            nn.Conv2d(channel_n * 3, 128, kernel_size=1),  # 48 → 128
            nn.ReLU(),
            nn.Conv2d(128, self.channel_n, kernel_size=1, bias=False)  # 128 → 16
        )
        
        # Initialize final layer to zero
        with torch.no_grad():
            self.brain[-1].weight.zero_()
```

### 5.6 Single Step Update

```python
def step(self, x, update_rate=None):
    # Perception
    y = perchannel_conv(x, filters)
    
    # Update Rule
    y = self.brain(y)

    # Stochastic cell update
    B, C, H, W = y.shape
    update_rate = update_rate or self.update_rate
    update_mask = (torch.rand(B, 1, H, W) + update_rate).floor()
    x = x + y * update_mask

    # Alive masking
    alive_mask = alive(x[:, 3:4, :, :], threshold=0.1)
    x = x * alive_mask

    return x
```

---

## 6. Training Experiments

### 6.1 Experiment 1: Learning to Grow

**Goal**: Train CA to reach target image from single seed cell.

**Setup**:
- Initialize grid with zeros except center seed (α and hidden channels = 1)
- Run 64-96 random steps per training iteration
- Apply L2 loss on RGBA channels

**Seed Initialization** (from `learning_to_grow.py`):
```python
seed = torch.zeros(1, 16, TARGET_SIZE, TARGET_SIZE)
seed[:, 3:, TARGET_SIZE//2, TARGET_SIZE//2] = 1.0  # α and hidden = 1
```

**Problem**: Models trained this way often become unstable when run beyond training steps - they may explode, die out, or oscillate.

### 6.2 Experiment 2: What Persists, Exists

**Goal**: Make the target pattern a stable attractor.

**Solution**: Pool-based training strategy.

```python
class SamplePool:
    def __init__(self, pool_size=1024, loss_fn=None):
        self.pool_size = pool_size
        self.loss_fn = loss_fn
        self.reset()
    
    def make_seed(self):
        seed = torch.zeros(1, 16, TARGET_SIZE, TARGET_SIZE)
        seed[:, 3:, TARGET_SIZE//2, TARGET_SIZE//2] = 1.0
        return seed

    def reset(self):
        self.pool = self.make_seed().repeat(self.pool_size, 1, 1, 1)

    def sample(self, num_samples=8):
        idxs = torch.randperm(self.pool_size)[:num_samples]
        batch = self.pool[idxs]
        
        # Replace highest-loss sample with seed (prevent forgetting)
        losses = self.loss_fn(batch[:, :4])
        replace_idx = torch.argmax(losses)
        batch[replace_idx] = self.seed[0]
        
        return batch
    
    def update(self, new_samples):
        self.pool[self.idxs] = new_samples.detach()
```

**Training Loop**:
1. Sample batch from pool
2. Replace worst sample with seed
3. Run CA for some steps
4. Compute loss & backpropagate
5. Replace pool samples with outputs

This forces the model to learn how to maintain/improve existing patterns, not just grow from seed.

### 6.3 Experiment 3: Learning to Regenerate

**Goal**: Robust regeneration from arbitrary damage.

**Key Addition**: Damage samples during training.

```python
def create_hole(batch):
    B, C, H, W = batch.shape
    
    # Create coordinate grid
    grid = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
    grid = torch.stack(grid, dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # Random hole center and radius
    center = torch.rand(B, 2, 1, 1) - 0.5      # [-0.5, 0.5]
    radius = 0.3 * torch.rand(B, 1, 1, 1) + 0.1  # [0.1, 0.4]
    
    # Create circular mask
    mask = ((grid - center)**2).sum(1, keepdim=True).sqrt() > radius
    
    return batch * mask.float()
```

**CorruptedPool Training**:
```python
class CorruptedPool(SamplePool):
    def sample_with_damage(self, num_samples=8, damaged_samples=3):
        batch = self.pool[idxs]
        
        # Sort by loss
        losses = self.loss_fn(batch[:, :4])
        sorted_idxs = torch.argsort(losses)
        
        # Replace worst with seed
        batch[sorted_idxs[-1]] = self.seed[0]
        
        # Damage the k best samples (lowest loss)
        damaged_idxs = sorted_idxs[:damaged_samples]
        batch[damaged_idxs] = create_hole(batch[damaged_idxs])
        
        return batch
```

**Result**: Models learn to regenerate from damage not seen during training (e.g., rectangular cuts).

### 6.4 Experiment 4: Rotating the Perceptive Field

**Insight**: Rotating Sobel filters produces rotated patterns without retraining.

$$
\begin{bmatrix} K_x \\ K_y \end{bmatrix} = 
\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
\begin{bmatrix} \text{Sobel}_x \\ \text{Sobel}_y \end{bmatrix}
$$

This works because the gradients are invariant to coordinate frame choice.

---

## 7. Code Reference

### File Structure

```
NeuralCellularAutomata_PyTorch/
├── core.py                 # Base model, filters, utilities
├── learning_to_grow.py     # Experiment 1: Seed initialization
├── what_persists_exists.py # Experiment 2: SamplePool
├── learning_to_regenerate.py # Experiment 3: CorruptedPool
```

### Key Functions

| Function | File | Description |
|----------|------|-------------|
| `perchannel_conv` | core.py | Applies Sobel filters per channel |
| `alive` | core.py | Computes alive mask via max pooling |
| `CAModel` | core.py | Main neural CA model |
| `display_animation` | core.py | Visualizes growth animation |
| `grow_animation` | core.py | Generates animation frames |
| `SamplePool` | what_persists_exists.py | Pool-based training |
| `create_hole` | learning_to_regenerate.py | Creates circular damage |
| `CorruptedPool` | learning_to_regenerate.py | Pool with damage sampling |

### Training Recipe

```python
# Initialize
ca = CAModel(16).to(device)
target = load_image('path/to/image.png')
pool = SamplePool(pool_size=1024, loss_fn=partial(mse, target=target))
optimizer = torch.optim.Adam(ca.parameters(), lr=2e-3)

# Training loop
for step in range(num_steps):
    batch = pool.sample(batch_size=8)
    
    # Random number of CA steps (64-96)
    n_steps = random.randint(64, 96)
    output = ca(batch, steps=n_steps)
    
    # Loss on RGBA channels only
    loss = mse(output[:, :4], target).mean()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient normalization (stabilizes training)
    for p in ca.parameters():
        p.grad.data = p.grad.data / (p.grad.data.norm() + 1e-8)
    
    optimizer.step()
    pool.update(output)
```

---

## References

- Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020). **Growing Neural Cellular Automata**. Distill. https://distill.pub/2020/growing-ca
- Original Colab Notebook: https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb

---

## Summary

The Growing Neural Cellular Automata model demonstrates that:

1. **Local rules can produce global behavior** - Each cell only sees its 3×3 neighborhood but collectively they build complex patterns
2. **Differentiable programming enables learning** - The update rule is learned via backpropagation
3. **Pool-based training creates attractors** - Reusing outputs as inputs teaches persistence
4. **Damage during training enables regeneration** - Models generalize to unseen damage types

The model uses only ~8,000 parameters and can run efficiently on GPUs or even be quantized for mobile deployment.
