# Particle Advection Refinement

## The Concept: "Painting with Flow"

The core idea is to treat the low-resolution NCA output not as a final image, but as a **map of forces**. We simulate thousands of tiny particles that are "dropped" onto this map. Instead of falling down (like gravity), they flow along the "contours" of the image features.

As these particles move, they carry the color of the area they spawned in. By drawing their paths, we reconstruct the image with high-resolution curves instead of blocky grid cells.

## Algorithm Steps

### 1. The "Terrain" (Preprocessing)
We start with the final $64 \times 64$ (or similar) grid from the NCA.
1.  **Upscale**: We resize this grid to your target render size (e.g., $1024 \times 1024$) using **Bicubic Interpolation**. This creates a smooth, blurry version of the image.
2.  **Grayscale Conversion**: We convert this color image to grayscale to determine "intensity".
3.  **Gradient Calculation**: We calculate the "slope" of the intensity at every pixel.
    *   Mathematically, we compute the derivatives $\frac{\partial I}{\partial x}$ and $\frac{\partial I}{\partial y}$ (using Sobel operators).
    *   This gives us a vector pointing "uphill" (from dark to light) at every point.

### 2. The Flow Field (The Math)
If particles followed the gradient, they would bunch up in the brightest areas. We want them to trace the **edges**.
To do this, we rotate the gradient vector by 90 degrees.

$$
\vec{V}_{flow} = \left( -\frac{\partial I}{\partial y}, \quad \frac{\partial I}{\partial x} \right)
$$

*   **Result**: A vector field where arrows point *along* the boundaries of shapes.
*   **Normalization**: We normalize these vectors so particles move at a constant speed, regardless of how sharp the edge is.

### 3. The Particle System
We initialize a system with $N$ particles (e.g., 20,000).
*   **Position**: Random $(x, y)$ coordinates.
*   **Color**: Sampled from the underlying NCA image at the particle's spawn location.
*   **Life**: Particles have a limited lifespan. When they die or move off-screen, they respawn at a new random location. This prevents them from getting stuck in "whirlpools".

### 4. The Animation Loop
For each frame of the animation:
1.  **Look up**: Each particle checks the Flow Field vector at its current integer coordinates.
2.  **Move**: The particle moves a tiny step in that direction.
3.  **Render**: We draw a pixel (or a small circle) at the particle's new position.

## Why this works for "Refinement"
Because the flow vectors are derived from the interpolated bicubic upscaling, the "blocky" pixel edges of the NCA are mathematically smoothed out into curves. The particles visualize these mathematical curves, effectively "hallucinating" smooth high-resolution details that connect the low-res blocks naturally.
