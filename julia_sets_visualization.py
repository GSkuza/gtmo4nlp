"""
GTMØ Julia Sets Visualization
==============================
Interactive visualization of Julia sets for different semantic attractors.
Each attractor has a unique fractal signature!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from IPython.display import HTML

# Julia parameters for each GTMØ attractor
JULIA_PARAMS = {
    "Ψᴷ (Knowledge)": complex(-0.4, 0.6),      # Stable, blob-like
    "Ψʰ (Shadow)": complex(0.285, 0.01),        # Disconnected dust
    "Ψᴺ (Emergent)": complex(-0.835, -0.2321),  # Dragon-like spirals
    "Ø (Singularity)": complex(-0.8, 0.156),    # Dendritic branches
    "Ψ~ (Flux)": complex(0.45, 0.1428),         # Swirling chaos
    "Ψ↑ (Transcendent)": complex(-0.70176, -0.3842), # Intricate spirals
    "Ψ◊ (Void)": complex(0.0, -0.8),            # Vertical symmetry
    "ℓ∅ (Alienated)": complex(-0.74543, 0.11301), # Organic tendrils
}

def compute_julia_set(c, width=800, height=800, x_min=-2, x_max=2, y_min=-2, y_max=2, max_iter=256):
    """
    Compute Julia set for given parameter c.
    Returns iteration counts for each point.
    """
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Initialize output array
    output = np.zeros(Z.shape, dtype=int)
    
    # Compute Julia set
    for i in range(height):
        for j in range(width):
            z = Z[i, j]
            for n in range(max_iter):
                if abs(z) > 2:
                    output[i, j] = n
                    break
                z = z * z + c
            else:
                output[i, j] = max_iter
    
    return output

def visualize_all_julia_sets():
    """
    Create a comprehensive visualization of all GTMØ Julia sets.
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('GTMØ Julia Sets - Fractal Signatures of Meaning Types', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Color maps for different attractors
    cmaps = ['hot', 'viridis', 'plasma', 'twilight', 'coolwarm', 'RdYlBu', 'copper', 'magma']
    
    # Generate and plot each Julia set
    for idx, (name, c) in enumerate(JULIA_PARAMS.items()):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Compute Julia set
        print(f"Computing {name}...")
        julia = compute_julia_set(c, width=400, height=400)
        
        # Apply logarithmic scaling for better visualization
        julia_log = np.log(julia + 1)
        
        # Plot
        im = ax.imshow(julia_log, extent=[-2, 2, -2, 2], 
                      cmap=cmaps[idx % len(cmaps)], 
                      origin='lower', interpolation='bilinear')
        
        # Formatting
        ax.set_title(f'{name}\nc = {c.real:.3f} + {c.imag:.3f}i', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Real', fontsize=9)
        ax.set_ylabel('Imaginary', fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Escape time (log)', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        # Mark origin
        ax.plot(0, 0, 'w+', markersize=8, markeredgewidth=1)
    
    # Add explanation box
    explanation_ax = fig.add_subplot(gs[2, 2:])
    explanation_ax.axis('off')
    
    explanation_text = """
    INTERPRETING JULIA FRACTALS:
    
    • COLORS: Escape speed (how fast points diverge)
      - Dark/Black = Stable (stays bounded)
      - Bright = Unstable (escapes quickly)
    
    • SHAPES reveal semantic nature:
      - Connected blob → Stable meanings (Ψᴷ)
      - Dust/fragments → Uncertain meanings (Ψʰ)  
      - Spirals → Emergent meanings (Ψᴺ)
      - Dendrites → Paradoxes (Ø)
    
    • BOUNDARIES: Fractal dimension shows
      complexity of meaning transitions
    """
    
    explanation_ax.text(0.05, 0.95, explanation_text, 
                       transform=explanation_ax.transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def zoom_into_julia(attractor_name="Ø (Singularity)", zoom_levels=5):
    """
    Animated zoom into a specific Julia set.
    Shows self-similarity at different scales.
    """
    c = JULIA_PARAMS[attractor_name]
    
    fig, axes = plt.subplots(1, zoom_levels, figsize=(20, 4))
    fig.suptitle(f'Zooming into {attractor_name} - Self-Similar Structure', 
                 fontsize=16, fontweight='bold')
    
    # Define zoom windows
    zoom_centers = [
        (0, 0),           # Full view
        (-0.5, 0.5),      # Zoom to interesting region
        (-0.55, 0.48),    # Further zoom
        (-0.553, 0.478),  # Even more
        (-0.5535, 0.4782) # Maximum zoom
    ]
    
    zoom_sizes = [2, 0.5, 0.1, 0.02, 0.004]
    
    for idx, (ax, center, size) in enumerate(zip(axes, zoom_centers, zoom_sizes)):
        # Compute Julia set for this zoom level
        julia = compute_julia_set(
            c, 
            width=300, 
            height=300,
            x_min=center[0] - size,
            x_max=center[0] + size,
            y_min=center[1] - size,
            y_max=center[1] + size,
            max_iter=256 + idx * 50  # Increase iterations for deeper zooms
        )
        
        # Plot
        im = ax.imshow(np.log(julia + 1), 
                      extent=[center[0]-size, center[0]+size, 
                             center[1]-size, center[1]+size],
                      cmap='hot', origin='lower')
        
        ax.set_title(f'Zoom {idx+1}: {size*2:.4f} units', fontsize=10)
        ax.set_xlabel('Real', fontsize=9)
        ax.set_ylabel('Imaginary', fontsize=9)
        
        # Mark zoom region for next level
        if idx < zoom_levels - 1:
            next_center = zoom_centers[idx + 1]
            next_size = zoom_sizes[idx + 1]
            rect = patches.Rectangle(
                (next_center[0] - next_size, next_center[1] - next_size),
                next_size * 2, next_size * 2,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    return fig

def compare_word_trajectories():
    """
    Show how different words behave in different Julia sets.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Word Trajectories in Different Julia Sets', 
                 fontsize=16, fontweight='bold')
    
    # Test words
    words = ["love", "mathematics", "paradox"]
    word_colors = ['red', 'blue', 'green']
    
    # Selected attractors
    selected_attractors = ["Ψᴷ (Knowledge)", "Ψʰ (Shadow)", "Ø (Singularity)"]
    
    for i, attractor_name in enumerate(selected_attractors):
        c = JULIA_PARAMS[attractor_name]
        
        # Top row: Julia set
        ax_julia = axes[0, i]
        julia = compute_julia_set(c, width=300, height=300)
        ax_julia.imshow(np.log(julia + 1), extent=[-2, 2, -2, 2], 
                       cmap='gray_r', origin='lower', alpha=0.7)
        ax_julia.set_title(f'{attractor_name}', fontsize=12, fontweight='bold')
        
        # Bottom row: Word trajectories
        ax_traj = axes[1, i]
        ax_traj.set_xlim(-2, 2)
        ax_traj.set_ylim(-2, 2)
        ax_traj.set_xlabel('Real')
        ax_traj.set_ylabel('Imaginary')
        ax_traj.grid(True, alpha=0.3)
        
        for word, color in zip(words, word_colors):
            # Convert word to complex number (simplified hash)
            word_hash = sum(ord(ch) for ch in word)
            z0 = complex(
                (word_hash % 100) / 50 - 1,  # Map to [-1, 1]
                (word_hash % 73) / 36.5 - 1
            )
            
            # Iterate through Julia function
            trajectory_x = [z0.real]
            trajectory_y = [z0.imag]
            z = z0
            
            for _ in range(50):
                z = z * z + c
                if abs(z) > 2:
                    break
                trajectory_x.append(z.real)
                trajectory_y.append(z.imag)
            
            # Plot trajectory
            ax_traj.plot(trajectory_x, trajectory_y, 
                        color=color, alpha=0.7, linewidth=1.5, label=word)
            ax_traj.plot(trajectory_x[0], trajectory_y[0], 
                        'o', color=color, markersize=8)  # Start point
            ax_traj.plot(trajectory_x[-1], trajectory_y[-1], 
                        's', color=color, markersize=8)  # End point
        
        ax_traj.legend(loc='upper right', fontsize=9)
        ax_traj.set_title(f'Word Iterations in {attractor_name.split()[0]}', fontsize=10)
    
    plt.tight_layout()
    return fig

# Main demonstration
def demonstrate_julia_sets():
    """
    Complete demonstration of Julia sets in GTMØ.
    """
    print("=" * 70)
    print("GTMØ JULIA SETS - FRACTAL SIGNATURES OF MEANING")
    print("=" * 70)
    
    print("\n1. GENERATING ALL JULIA SETS...")
    print("-" * 40)
    fig1 = visualize_all_julia_sets()
    plt.show()
    
    print("\n2. ZOOMING INTO SINGULARITY (Ø)...")
    print("-" * 40)
    print("Notice how the pattern repeats at different scales!")
    fig2 = zoom_into_julia("Ø (Singularity)", zoom_levels=5)
    plt.show()
    
    print("\n3. WORD TRAJECTORY ANALYSIS...")
    print("-" * 40)
    print("Different words follow different paths in each attractor's Julia set:")
    fig3 = compare_word_trajectories()
    plt.show()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("-" * 40)
    print("• Each attractor has a unique fractal 'fingerprint'")
    print("• Stable meanings (Ψᴷ) → Connected, blob-like Julia sets")
    print("• Uncertain meanings (Ψʰ) → Fragmented, dust-like sets")
    print("• Paradoxes (Ø) → Dendritic, branching structures")
    print("• The boundaries are infinitely complex (fractal dimension ~1.5)")
    print("• Words iterate differently through each attractor's fractal")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_julia_sets()