"""
GTMØ Fractal Geometry of Meanings
==================================
Implementation of fractal semantic basins in GTMØ theory.
Focus on iterative mapping, fractal boundaries, and emergence patterns.

Key features:
- Fractal basin boundaries between semantic attractors
- Iterative function system (IFS) for meaning evolution
- Julia/Mandelbrot-like sets in semantic space
- Strange attractors and chaotic trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# ============================================================================
# FRACTAL CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio for self-similarity
FEIGENBAUM_DELTA = 4.669  # Period-doubling constant
HAUSDORFF_DIM = 1.585  # Fractal dimension estimate for semantic boundaries

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SemanticPoint:
    """Point in semantic fractal space."""
    d: float  # Determination
    s: float  # Stability 
    e: float  # Entropy
    text: str = ""
    iteration: int = 0
    
    def to_complex(self) -> complex:
        """Map to complex plane for fractal operations."""
        return complex(self.d - 0.5, self.s - 0.5)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.d, self.s, self.e])

@dataclass
class FractalAttractor:
    """Attractor with fractal basin properties."""
    name: str
    center: np.ndarray
    strength: float
    julia_c: complex  # Julia set parameter for this attractor
    
    def basin_function(self, z: complex, c: complex = None) -> complex:
        """Julia set iteration for this attractor's basin."""
        if c is None:
            c = self.julia_c
        return z**2 + c

# ============================================================================
# FRACTAL ATTRACTORS
# ============================================================================

FRACTAL_ATTRACTORS = [
    FractalAttractor("Ψᴷ", np.array([0.85, 0.85, 0.15]), 1.0, complex(-0.4, 0.6)),
    FractalAttractor("Ψʰ", np.array([0.15, 0.15, 0.85]), 1.0, complex(0.285, 0.01)),
    FractalAttractor("Ψᴺ", np.array([0.50, 0.30, 0.90]), 1.2, complex(-0.835, -0.2321)),
    FractalAttractor("Ø", np.array([1.00, 1.00, 0.00]), 2.0, complex(-0.8, 0.156)),
    FractalAttractor("Ψ~", np.array([0.50, 0.50, 0.80]), 0.9, complex(0.45, 0.1428)),
]

# ============================================================================
# FRACTAL OPERATIONS
# ============================================================================

class FractalSemanticSpace:
    """Fractal geometry engine for semantic meanings."""
    
    def __init__(self):
        self.attractors = FRACTAL_ATTRACTORS
        self.iteration_depth = 100
        self.escape_radius = 2.0
        
    def iterate_meaning(self, point: SemanticPoint, 
                       attractor: FractalAttractor, 
                       steps: int = 10) -> List[SemanticPoint]:
        """
        Iterate a semantic point through fractal transformation.
        Creates a trajectory showing semantic evolution.
        """
        trajectory = [point]
        z = point.to_complex()
        
        for i in range(steps):
            # Apply fractal iteration
            z = attractor.basin_function(z)
            
            # Add noise for semantic drift
            noise = complex(np.random.randn() * 0.01, np.random.randn() * 0.01)
            z += noise
            
            # Convert back to semantic coordinates
            d = abs(z.real) % 1
            s = abs(z.imag) % 1
            e = (abs(z) / self.escape_radius) % 1
            
            new_point = SemanticPoint(d, s, e, f"{point.text}_iter{i+1}", i+1)
            trajectory.append(new_point)
            
            # Check for escape to infinity (semantic dissolution)
            if abs(z) > self.escape_radius:
                break
                
        return trajectory
    
    def calculate_fractal_dimension(self, points: List[SemanticPoint], 
                                   epsilon_range: np.ndarray = None) -> float:
        """
        Estimate Hausdorff dimension of semantic trajectory.
        Uses box-counting method.
        """
        if epsilon_range is None:
            epsilon_range = np.logspace(-3, 0, 20)
        
        coords = np.array([p.to_array()[:2] for p in points])  # Use D,S dimensions
        
        counts = []
        for epsilon in epsilon_range:
            # Count boxes needed to cover trajectory
            boxes = set()
            for coord in coords:
                box = tuple((coord // epsilon).astype(int))
                boxes.add(box)
            counts.append(len(boxes))
        
        # Linear regression in log-log space
        log_eps = np.log(epsilon_range)
        log_counts = np.log(counts)
        
        # Fractal dimension is negative slope
        coeffs = np.polyfit(log_eps, log_counts, 1)
        return -coeffs[0]
    
    def generate_julia_basin(self, resolution: int = 200,
                            attractor_idx: int = 0,
                            x_range: Tuple[float, float] = (-2, 2),
                            y_range: Tuple[float, float] = (-2, 2)) -> np.ndarray:
        """
        Generate Julia set for semantic attractor basin.
        Shows fractal structure of meaning boundaries.
        """
        attractor = self.attractors[attractor_idx]
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        # Julia set iteration
        basin = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                z = Z[i, j]
                for k in range(self.iteration_depth):
                    z = attractor.basin_function(z)
                    if abs(z) > self.escape_radius:
                        basin[i, j] = k / self.iteration_depth
                        break
                else:
                    basin[i, j] = 1.0  # Didn't escape - in the set
                    
        return basin
    
    def semantic_ifs(self, initial_text: str, 
                    transformations: List[Callable],
                    iterations: int = 1000) -> np.ndarray:
        """
        Iterated Function System for semantic fractals.
        Generates Sierpinski-like patterns in meaning space.
        """
        # Start from text projection
        point = self.project_text(initial_text)
        points = []
        
        for _ in range(iterations):
            # Randomly choose transformation
            transform = np.random.choice(transformations)
            point = transform(point)
            points.append(point.to_array()[:2])  # Store D,S coordinates
            
        return np.array(points)
    
    def project_text(self, text: str) -> SemanticPoint:
        """Project text to semantic fractal space."""
        # Hash-based deterministic projection
        hash_val = hash(text)
        
        # Use golden ratio for better distribution
        d = (hash_val * PHI) % 1
        s = (hash_val * PHI**2) % 1
        e = (hash_val * PHI**3) % 1
        
        return SemanticPoint(d, s, e, text)
    
    def detect_strange_attractor(self, trajectory: List[SemanticPoint]) -> dict:
        """
        Detect if trajectory forms a strange attractor.
        Checks for chaos indicators.
        """
        if len(trajectory) < 10:
            return {"is_strange": False, "reason": "Too few points"}
        
        coords = np.array([p.to_array() for p in trajectory])
        
        # Check for bounded but non-periodic behavior
        distances = []
        for i in range(1, len(coords)):
            distances.append(np.linalg.norm(coords[i] - coords[i-1]))
        
        # Calculate Lyapunov exponent estimate
        lyapunov = np.mean(np.log(np.abs(np.diff(distances) + 1e-10)))
        
        # Check phase space coverage
        coverage = len(np.unique(coords.round(decimals=2), axis=0)) / len(coords)
        
        is_strange = lyapunov > 0 and coverage > 0.5
        
        return {
            "is_strange": is_strange,
            "lyapunov": lyapunov,
            "coverage": coverage,
            "dimension": self.calculate_fractal_dimension(trajectory) if is_strange else None
        }
    
    def visualize_fractal_basins(self, resolution: int = 400):
        """
        Visualize fractal basins of all attractors.
        Creates a map showing which attractor dominates each region.
        """
        # Create grid
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate basin for each point
        basin_map = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                point = SemanticPoint(X[i, j], Y[i, j], 0.5)
                
                # Iterate and see which attractor it approaches
                z = point.to_complex()
                for _ in range(50):
                    min_dist = float('inf')
                    nearest_idx = 0
                    
                    # Find nearest attractor
                    for k, att in enumerate(self.attractors):
                        z_test = att.basin_function(z)
                        dist = abs(z_test - complex(att.center[0]-0.5, att.center[1]-0.5))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = k
                    
                    z = self.attractors[nearest_idx].basin_function(z)
                    
                    if abs(z) > self.escape_radius:
                        basin_map[i, j] = -1  # Escaped
                        break
                else:
                    basin_map[i, j] = nearest_idx
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Basin map
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'black']
        cmap = ListedColormap(colors[:len(self.attractors)+1])
        
        im1 = ax1.imshow(basin_map.T, extent=[0, 1, 0, 1], origin='lower', cmap=cmap)
        ax1.set_title('Fractal Semantic Basins', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Determination →')
        ax1.set_ylabel('Stability →')
        
        # Mark attractors
        for i, att in enumerate(self.attractors):
            ax1.plot(att.center[0], att.center[1], 'w*', markersize=15, 
                    markeredgecolor='black', markeredgewidth=2)
            ax1.text(att.center[0], att.center[1]-0.05, att.name, 
                    ha='center', fontsize=10, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Julia set for first attractor
        julia = self.generate_julia_basin(resolution=200, attractor_idx=0)
        im2 = ax2.imshow(julia, extent=[-2, 2, -2, 2], cmap='hot', origin='lower')
        ax2.set_title(f'Julia Set for {self.attractors[0].name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Real →')
        ax2.set_ylabel('Imaginary →')
        
        plt.colorbar(im2, ax=ax2, label='Escape time')
        plt.tight_layout()
        plt.show()
        
        return basin_map

# ============================================================================
# IFS TRANSFORMATIONS
# ============================================================================

def sierpinski_semantic_transform():
    """Create Sierpinski-like transformations for semantic space."""
    
    def t1(p: SemanticPoint) -> SemanticPoint:
        return SemanticPoint(p.d * 0.5, p.s * 0.5, p.e, p.text + "_t1")
    
    def t2(p: SemanticPoint) -> SemanticPoint:
        return SemanticPoint(p.d * 0.5 + 0.5, p.s * 0.5, p.e, p.text + "_t2")
    
    def t3(p: SemanticPoint) -> SemanticPoint:
        return SemanticPoint(p.d * 0.5 + 0.25, p.s * 0.5 + 0.5, p.e, p.text + "_t3")
    
    return [t1, t2, t3]

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_fractal_semantics():
    """Demonstrate fractal geometry of meanings."""
    print("=" * 70)
    print("GTMØ FRACTAL GEOMETRY OF MEANINGS")
    print("=" * 70)
    
    # Initialize fractal space
    fractal = FractalSemanticSpace()
    
    # Test 1: Fractal iteration
    print("\n1. FRACTAL ITERATION OF MEANING:")
    print("-" * 40)
    
    initial = fractal.project_text("consciousness")
    trajectory = fractal.iterate_meaning(initial, fractal.attractors[2], steps=15)
    
    print(f"Initial: D={initial.d:.3f}, S={initial.s:.3f}, E={initial.e:.3f}")
    for i, point in enumerate(trajectory[1:6], 1):
        print(f"Iter {i}: D={point.d:.3f}, S={point.s:.3f}, E={point.e:.3f}")
    
    # Calculate fractal dimension
    if len(trajectory) > 5:
        dim = fractal.calculate_fractal_dimension(trajectory)
        print(f"\nFractal dimension of trajectory: {dim:.3f}")
    
    # Test 2: Strange attractor detection
    print("\n2. STRANGE ATTRACTOR DETECTION:")
    print("-" * 40)
    
    # Create chaotic trajectory
    chaotic = fractal.project_text("paradox")
    chaotic_traj = fractal.iterate_meaning(chaotic, fractal.attractors[3], steps=50)
    
    strange_analysis = fractal.detect_strange_attractor(chaotic_traj)
    print(f"Is strange attractor: {strange_analysis['is_strange']}")
    if strange_analysis['is_strange']:
        print(f"Lyapunov exponent: {strange_analysis['lyapunov']:.3f}")
        print(f"Phase space coverage: {strange_analysis['coverage']:.1%}")
        print(f"Fractal dimension: {strange_analysis['dimension']:.3f}")
    
    # Test 3: IFS fractals
    print("\n3. ITERATED FUNCTION SYSTEM:")
    print("-" * 40)
    
    transforms = sierpinski_semantic_transform()
    ifs_points = fractal.semantic_ifs("language", transforms, iterations=1000)
    
    print(f"Generated {len(ifs_points)} IFS points")
    print(f"Bounding box: [{ifs_points.min():.3f}, {ifs_points.max():.3f}]")
    
    # Visualize IFS
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ifs_points[:, 0], ifs_points[:, 1], s=0.5, alpha=0.5, c='blue')
    ax.set_title('Semantic IFS Fractal (Sierpinski-like)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Determination →')
    ax.set_ylabel('Stability →')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Test 4: Fractal basins visualization
    print("\n4. FRACTAL BASIN VISUALIZATION:")
    print("-" * 40)
    print("Generating fractal basin map...")
    
    basin_map = fractal.visualize_fractal_basins(resolution=300)
    
    unique_basins = len(np.unique(basin_map))
    print(f"Found {unique_basins} distinct basins")
    
    # Calculate basin boundary dimension
    boundary_points = []
    for i in range(1, basin_map.shape[0]-1):
        for j in range(1, basin_map.shape[1]-1):
            neighbors = [
                basin_map[i-1, j], basin_map[i+1, j],
                basin_map[i, j-1], basin_map[i, j+1]
            ]
            if len(set(neighbors)) > 1:  # Boundary point
                boundary_points.append(SemanticPoint(i/basin_map.shape[0], 
                                                    j/basin_map.shape[1], 
                                                    0.5))
    
    if boundary_points:
        boundary_dim = fractal.calculate_fractal_dimension(boundary_points[:100])
        print(f"Fractal dimension of basin boundaries: {boundary_dim:.3f}")
    
    print("\n" + "=" * 70)
    print("FRACTAL SEMANTICS COMPLETE")
    print("The boundaries between meanings are fractal!")
    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demonstrate_fractal_semantics()