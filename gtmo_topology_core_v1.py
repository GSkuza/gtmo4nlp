"""
gtmo_topology_core_v1.py
========================
Core implementation of GTMØ (Geometry Topology Mathematics Ø) theory.

This module provides a clean, well-documented implementation of the GTMØ framework
for analyzing linguistic subjectivity through mathematical topology. It includes
phase space analysis, topological attractors, and visualization capabilities.

Author: Based on GTMØ theory by Grzegorz Skuza
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import warnings

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not found. Visualization functions will be disabled.")


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

# Core GTMØ constants based on the theory
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
SQRT_2_INV = 1 / np.sqrt(2)  # Quantum superposition amplitude ≈ 0.707
COGNITIVE_CENTER = np.array([0.5, 0.5, 0.5])  # Neutral knowledge state
BOUNDARY_THICKNESS = 0.02  # Epistemic boundary thickness
ENTROPY_THRESHOLD = 0.001  # Threshold for singularity collapse
BREATHING_AMPLITUDE = 0.1  # Cognitive space pulsation amplitude


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class KnowledgeType(Enum):
    """Enumeration of epistemic knowledge types in GTMØ theory."""
    PARTICLE = "Ψᴷ"      # Clear, stable information
    SHADOW = "Ψʰ"        # Unclear, unstable information
    EMERGENT = "Ψᴺ"      # Information creating new meanings
    LIMINALITY = "Ψˡ"    # Boundary state, transitional
    SINGULARITY = "Ø"    # Paradoxes, logical contradictions
    TRANSCENDENT = "Ψ↑"  # Transcendent knowledge
    FLUX = "Ψ~"          # Fluid, changing knowledge
    VOID = "Ψ◊"          # Void or absent knowledge
    ALIENATED = "ℓ∅"     # Alienated numbers (hybrid meanings)


@dataclass
class TopologicalAttractor:
    """
    Represents a topological attractor in the GTMØ phase space.
    
    Attributes:
        name: Name of the attractor
        knowledge_type: Type of knowledge this attractor represents
        position: 3D coordinates [determination, stability, entropy]
        radius: Basin of attraction radius
        strength: Attraction strength (higher = stronger pull)
    """
    name: str
    knowledge_type: KnowledgeType
    position: np.ndarray
    radius: float
    strength: float
    
    def __post_init__(self):
        """Validate attractor parameters."""
        self.position = np.array(self.position)
        assert self.position.shape == (3,), "Position must be 3D"
        assert 0 <= self.radius <= 1, "Radius must be in [0, 1]"
        assert self.strength > 0, "Strength must be positive"


@dataclass
class PhasePoint:
    """
    Represents a point in the GTMØ phase space.
    
    Attributes:
        determination: How unambiguous/clear the meaning is (0-1)
        stability: How constant the meaning is over time (0-1)
        entropy: How chaotic/creative the meaning is (0-1)
        label: Optional label for the point
    """
    determination: float
    stability: float
    entropy: float
    label: Optional[str] = None
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array representation."""
        return np.array([self.determination, self.stability, self.entropy])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, label: Optional[str] = None) -> 'PhasePoint':
        """Create PhasePoint from numpy array."""
        assert arr.shape == (3,), "Array must be 3D"
        return cls(arr[0], arr[1], arr[2], label)


# =============================================================================
# ATTRACTOR DEFINITIONS
# =============================================================================

# Define the topological attractors based on GTMØ theory
ATTRACTORS = [
    TopologicalAttractor(
        name="Singularity",
        knowledge_type=KnowledgeType.SINGULARITY,
        position=np.array([1.0, 1.0, 0.0]),
        radius=0.15,
        strength=2.0
    ),
    TopologicalAttractor(
        name="Knowledge Particle",
        knowledge_type=KnowledgeType.PARTICLE,
        position=np.array([0.85, 0.85, 0.15]),
        radius=0.25,
        strength=1.0
    ),
    TopologicalAttractor(
        name="Knowledge Shadow",
        knowledge_type=KnowledgeType.SHADOW,
        position=np.array([0.15, 0.15, 0.85]),
        radius=0.25,
        strength=1.0
    ),
    TopologicalAttractor(
        name="Emergent",
        knowledge_type=KnowledgeType.EMERGENT,
        position=np.array([0.5, 0.3, 0.9]),
        radius=0.20,
        strength=1.2
    ),
    TopologicalAttractor(
        name="Transcendent",
        knowledge_type=KnowledgeType.TRANSCENDENT,
        position=np.array([0.7, 0.7, 0.3]),
        radius=0.15,
        strength=1.1
    ),
    TopologicalAttractor(
        name="Flux",
        knowledge_type=KnowledgeType.FLUX,
        position=np.array([0.5, 0.5, 0.8]),
        radius=0.30,
        strength=0.9
    ),
    TopologicalAttractor(
        name="Void",
        knowledge_type=KnowledgeType.VOID,
        position=np.array([0.0, 0.0, 0.5]),
        radius=0.20,
        strength=0.8
    ),
    TopologicalAttractor(
        name="Alienated",
        knowledge_type=KnowledgeType.ALIENATED,
        position=np.array([0.999, 0.999, 0.001]),
        radius=0.10,
        strength=1.5
    )
]


# =============================================================================
# CORE GTMØ SYSTEM
# =============================================================================

class GTMOTopologyCore:
    """
    Core implementation of the GTMØ topology system.
    
    This class provides methods for:
    - Phase space analysis
    - Topological classification
    - Distance calculations
    - Trajectory analysis
    - Visualization
    """
    
    def __init__(self, attractors: Optional[List[TopologicalAttractor]] = None):
        """
        Initialize the GTMØ topology system.
        
        Args:
            attractors: List of topological attractors (defaults to GTMØ standard set)
        """
        self.attractors = attractors or ATTRACTORS
        self.cognitive_center = COGNITIVE_CENTER.copy()
        self.breathing_phase = 0.0  # Phase for cognitive space pulsation
        
    def calculate_distance(self, point: Union[PhasePoint, np.ndarray], 
                         attractor: TopologicalAttractor) -> float:
        """
        Calculate the effective distance from a point to an attractor.
        
        Uses Euclidean distance weighted by attractor strength.
        
        Args:
            point: Point in phase space
            attractor: Target attractor
            
        Returns:
            Effective distance (lower = closer/stronger attraction)
        """
        if isinstance(point, PhasePoint):
            point = point.to_array()
        
        euclidean_dist = np.linalg.norm(point - attractor.position)
        effective_dist = euclidean_dist / attractor.strength
        
        return effective_dist
    
    def classify_point(self, point: Union[PhasePoint, np.ndarray]) -> Tuple[KnowledgeType, float]:
        """
        Classify a point in phase space by finding the nearest attractor.
        
        Args:
            point: Point to classify
            
        Returns:
            Tuple of (knowledge_type, confidence)
            where confidence = 1 - (distance_to_nearest / radius_of_nearest)
        """
        if isinstance(point, PhasePoint):
            point = point.to_array()
        
        min_distance = float('inf')
        nearest_attractor = None
        
        for attractor in self.attractors:
            distance = self.calculate_distance(point, attractor)
            if distance < min_distance:
                min_distance = distance
                nearest_attractor = attractor
        
        # Calculate confidence based on how deep the point is within the basin
        confidence = max(0, 1 - (min_distance / nearest_attractor.radius))
        
        return nearest_attractor.knowledge_type, confidence
    
    def analyze_trajectory(self, trajectory: List[PhasePoint]) -> Dict:
        """
        Analyze a trajectory through phase space.
        
        Args:
            trajectory: List of phase points representing a path
            
        Returns:
            Dictionary containing trajectory metrics
        """
        if len(trajectory) < 2:
            return {"error": "Trajectory must contain at least 2 points"}
        
        # Convert to numpy array for easier computation
        points = np.array([p.to_array() for p in trajectory])
        
        # Calculate trajectory metrics
        total_distance = 0
        for i in range(1, len(points)):
            total_distance += np.linalg.norm(points[i] - points[i-1])
        
        # Analyze dominant attractors along the path
        classifications = [self.classify_point(p) for p in points]
        dominant_types = {}
        for k_type, conf in classifications:
            if k_type not in dominant_types:
                dominant_types[k_type] = 0
            dominant_types[k_type] += conf
        
        # Find trajectory bounds
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        return {
            "total_distance": total_distance,
            "average_step": total_distance / (len(trajectory) - 1),
            "dominant_types": dominant_types,
            "start_classification": classifications[0],
            "end_classification": classifications[-1],
            "bounds": {
                "min": min_coords.tolist(),
                "max": max_coords.tolist()
            },
            "center_of_mass": np.mean(points, axis=0).tolist()
        }
    
    def breathe(self, delta_phase: float = 0.1) -> None:
        """
        Update the breathing phase of cognitive space.
        
        This models the dynamic nature of knowledge boundaries.
        
        Args:
            delta_phase: Phase increment (radians)
        """
        self.breathing_phase += delta_phase
        breathing_factor = 1 + BREATHING_AMPLITUDE * np.sin(self.breathing_phase)
        
        # Update attractor radii based on breathing
        for attractor in self.attractors:
            base_radius = attractor.radius / (1 + BREATHING_AMPLITUDE)
            attractor.radius = base_radius * breathing_factor
    
    def visualize_phase_space(self, points: Optional[List[PhasePoint]] = None,
                            show_attractors: bool = True,
                            show_trajectories: bool = False) -> None:
        """
        Create a 3D visualization of the phase space.
        
        Args:
            points: Optional list of points to plot
            show_attractors: Whether to show attractor positions
            show_trajectories: Whether to connect points as trajectory
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot visualize: Matplotlib is not installed.")
            return
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot attractors
        if show_attractors:
            for attractor in self.attractors:
                pos = attractor.position
                ax.scatter(pos[0], pos[1], pos[2], 
                          s=200 * attractor.strength, 
                          marker='*',
                          label=f"{attractor.name} ({attractor.knowledge_type.value})",
                          alpha=0.8)
                
                # Draw attractor basin (simplified as sphere)
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = pos[0] + attractor.radius * np.outer(np.cos(u), np.sin(v))
                y = pos[1] + attractor.radius * np.outer(np.sin(u), np.sin(v))
                z = pos[2] + attractor.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.1)
        
        # Plot points
        if points:
            coords = np.array([p.to_array() for p in points])
            colors = coords[:, 2]  # Color by entropy
            
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               c=colors, cmap='viridis', s=50, alpha=0.7)
            
            if show_trajectories and len(points) > 1:
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                       'k-', alpha=0.3, linewidth=1)
            
            plt.colorbar(scatter, ax=ax, label='Entropy')
        
        # Plot cognitive center
        ax.scatter(*self.cognitive_center, s=100, c='red', marker='o',
                  label='Cognitive Center')
        
        # Set labels and title
        ax.set_xlabel('Determination')
        ax.set_ylabel('Stability')
        ax.set_zlabel('Entropy')
        ax.set_title('GTMØ Phase Space Topology')
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_semantic_trajectory(word: str, time_points: List[Tuple[float, float, float]], 
                             labels: Optional[List[str]] = None) -> List[PhasePoint]:
    """
    Create a semantic trajectory for a word through phase space.
    
    Args:
        word: The word being traced
        time_points: List of (determination, stability, entropy) tuples
        labels: Optional labels for each time point
        
    Returns:
        List of PhasePoint objects representing the trajectory
    """
    trajectory = []
    for i, (d, s, e) in enumerate(time_points):
        label = f"{word}_{labels[i]}" if labels and i < len(labels) else f"{word}_t{i}"
        trajectory.append(PhasePoint(d, s, e, label))
    
    return trajectory


def demonstrate_system():
    """Demonstrate the GTMØ topology system with examples."""
    print("=" * 80)
    print("GTMØ TOPOLOGY CORE v1.0 - DEMONSTRATION")
    print("=" * 80)
    
    # Initialize system
    system = GTMOTopologyCore()
    
    # Example 1: Classify individual concepts
    print("\n1. CLASSIFYING INDIVIDUAL CONCEPTS:")
    print("-" * 40)
    
    test_points = [
        PhasePoint(0.95, 0.95, 0.05, "Mathematical fact: 2+2=4"),
        PhasePoint(0.3, 0.4, 0.6, "Weather prediction"),
        PhasePoint(0.1, 0.1, 0.95, "Paradox: This sentence is false"),
        PhasePoint(0.5, 0.3, 0.9, "Viral meme"),
        PhasePoint(0.7, 0.7, 0.3, "Scientific theory")
    ]
    
    for point in test_points:
        k_type, confidence = system.classify_point(point)
        print(f"{point.label}:")
        print(f"  → Type: {k_type.value} ({k_type.name})")
        print(f"  → Confidence: {confidence:.2%}")
    
    # Example 2: Analyze word trajectory
    print("\n\n2. ANALYZING WORD TRAJECTORY - 'VIRUS':")
    print("-" * 40)
    
    virus_trajectory = create_semantic_trajectory(
        "virus",
        [
            (0.2, 0.8, 0.1),   # 1900: Unknown concept
            (0.8, 0.9, 0.2),   # 1950: Clear biological definition
            (0.6, 0.5, 0.7),   # 1990: Computer virus emerges
            (0.7, 0.3, 0.9)    # 2020: COVID + viral content
        ],
        ["1900", "1950", "1990", "2020"]
    )
    
    analysis = system.analyze_trajectory(virus_trajectory)
    print(f"Total trajectory distance: {analysis['total_distance']:.3f}")
    print(f"Average step size: {analysis['average_step']:.3f}")
    print(f"Start: {analysis['start_classification'][0].value} "
          f"({analysis['start_classification'][1]:.2%} confidence)")
    print(f"End: {analysis['end_classification'][0].value} "
          f"({analysis['end_classification'][1]:.2%} confidence)")
    
    # Example 3: Visualization
    if MATPLOTLIB_AVAILABLE:
        print("\n\n3. GENERATING PHASE SPACE VISUALIZATION...")
        print("-" * 40)
        
        # Create sample points across different knowledge types
        sample_points = []
        np.random.seed(42)
        
        # Add clustered points near each attractor
        for attractor in system.attractors[:4]:  # Use first 4 attractors
            for _ in range(10):
                # Generate points within attractor basin
                offset = np.random.randn(3) * attractor.radius * 0.3
                point_coords = np.clip(attractor.position + offset, 0, 1)
                sample_points.append(PhasePoint.from_array(point_coords))
        
        # Add the virus trajectory
        sample_points.extend(virus_trajectory)
        
        # Visualize
        system.visualize_phase_space(
            points=sample_points,
            show_attractors=True,
            show_trajectories=False
        )
    
    # Example 4: Breathing cognitive space
    print("\n4. COGNITIVE SPACE BREATHING:")
    print("-" * 40)
    
    initial_radius = system.attractors[0].radius
    for i in range(5):
        system.breathe(np.pi / 4)  # Breathe by π/4 radians
        new_radius = system.attractors[0].radius
        print(f"Step {i+1}: Radius changed from {initial_radius:.3f} to {new_radius:.3f}")
        initial_radius = new_radius


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    demonstrate_system()
