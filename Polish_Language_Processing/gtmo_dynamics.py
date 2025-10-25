#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Dynamics - Advanced Semantic Dynamics Analysis
====================================================

Implements:
- Hamiltonian dynamics for semantic energy evolution
- Julia set iterations for modeling semantic emergence
- Contextual dynamics for temporal evolution
- Trajectory visualization in GTMØ phase space

Author: GTMØ Framework
Date: 2025-10-25
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  matplotlib not available - visualization disabled")


@dataclass
class GTMOCoordinates:
    """Współrzędne w przestrzeni fazowej GTMØ."""
    determination: float  # D ∈ [0,1]
    stability: float      # S ∈ [0,1]
    entropy: float        # E ∈ [0,1]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for calculations."""
        return np.array([self.determination, self.stability, self.entropy])

    def distance_to(self, other: 'GTMOCoordinates') -> float:
        """Calculate Euclidean distance to another point."""
        return np.linalg.norm(self.to_array() - other.to_array())

    def __repr__(self) -> str:
        return f"GTMØ(D={self.determination:.3f}, S={self.stability:.3f}, E={self.entropy:.3f})"


@dataclass
class SemanticTrajectory:
    """Trajektoria semantyczna w przestrzeni GTMØ."""
    points: List[GTMOCoordinates]
    timestamps: List[float]
    energies: List[float]
    contexts: List[str]

    def length(self) -> float:
        """Calculate total trajectory length."""
        if len(self.points) < 2:
            return 0.0
        return sum(self.points[i].distance_to(self.points[i+1])
                   for i in range(len(self.points)-1))

    def mean_energy(self) -> float:
        """Calculate mean energy along trajectory."""
        return np.mean(self.energies) if self.energies else 0.0


class SemanticHamiltonian:
    """
    Hamiltonian dynamics for semantic evolution in GTMØ space.

    Models semantic energy as:
    H = K + V = (p²/2m) + V(q)

    where:
    - K: kinetic energy (semantic momentum)
    - V: potential energy (semantic configuration)
    - q: position in GTMØ space (D, S, E)
    - p: momentum conjugate to q
    """

    def __init__(self, mass: float = 1.0, dt: float = 0.01):
        """
        Initialize Hamiltonian system.

        Args:
            mass: Mass parameter for kinetic energy
            dt: Time step for integration
        """
        self.mass = mass
        self.dt = dt
        self.trajectory_history = []

    def potential_energy(self, coords: GTMOCoordinates) -> float:
        """
        Calculate potential energy at given coordinates.

        V(D, S, E) = α·D² + β·S² + γ·E² + δ·D·S·E

        Represents semantic "landscape" with:
        - High D, high S: deep well (stable knowledge)
        - High E: energy barrier (undefined states)
        """
        D, S, E = coords.determination, coords.stability, coords.entropy

        # Quadratic terms
        alpha, beta, gamma = 0.5, 0.5, 1.0
        quadratic = alpha * D**2 + beta * S**2 + gamma * E**2

        # Interaction term (coupling between dimensions)
        delta = -0.3
        interaction = delta * D * S * E

        # Entropic barrier (exponential repulsion at high entropy)
        epsilon = 0.2
        barrier = epsilon * np.exp(5 * (E - 0.5)) if E > 0.5 else 0.0

        return quadratic + interaction + barrier

    def kinetic_energy(self, momentum: np.ndarray) -> float:
        """
        Calculate kinetic energy from momentum.

        K = p²/(2m)
        """
        return np.sum(momentum**2) / (2 * self.mass)

    def total_energy(self, coords: GTMOCoordinates, momentum: np.ndarray) -> float:
        """Calculate total Hamiltonian H = K + V."""
        return self.kinetic_energy(momentum) + self.potential_energy(coords)

    def force(self, coords: GTMOCoordinates) -> np.ndarray:
        """
        Calculate force as negative gradient of potential.

        F = -∇V
        """
        # Numerical gradient using finite differences
        epsilon = 1e-6
        gradient = np.zeros(3)

        base_energy = self.potential_energy(coords)

        for i in range(3):
            coords_array = coords.to_array()
            coords_array[i] += epsilon

            # Create perturbed coordinates
            perturbed = GTMOCoordinates(*np.clip(coords_array, 0, 1))
            gradient[i] = (self.potential_energy(perturbed) - base_energy) / epsilon

        return -gradient

    def evolve(self,
               initial_coords: GTMOCoordinates,
               initial_momentum: Optional[np.ndarray] = None,
               steps: int = 100) -> SemanticTrajectory:
        """
        Evolve system using Hamiltonian dynamics (Verlet integration).

        Args:
            initial_coords: Starting position in GTMØ space
            initial_momentum: Starting momentum (random if None)
            steps: Number of time steps

        Returns:
            SemanticTrajectory with evolution history
        """
        if initial_momentum is None:
            # Random initial momentum
            initial_momentum = np.random.randn(3) * 0.1

        # Initialize arrays
        q = initial_coords.to_array().copy()
        p = initial_momentum.copy()

        points = [GTMOCoordinates(*q)]
        energies = [self.total_energy(points[0], p)]
        timestamps = [0.0]

        # Verlet integration
        for step in range(1, steps):
            # Update position: q(t+dt) = q(t) + p(t)/m * dt
            q = q + (p / self.mass) * self.dt

            # Clip to valid range [0, 1]
            q = np.clip(q, 0, 1)

            # Create current coordinates
            coords = GTMOCoordinates(*q)

            # Update momentum: p(t+dt) = p(t) + F(q) * dt
            F = self.force(coords)
            p = p + F * self.dt

            # Record trajectory
            points.append(coords)
            energies.append(self.total_energy(coords, p))
            timestamps.append(step * self.dt)

        trajectory = SemanticTrajectory(
            points=points,
            timestamps=timestamps,
            energies=energies,
            contexts=[f"t={t:.2f}" for t in timestamps]
        )

        self.trajectory_history.append(trajectory)
        return trajectory

    def find_equilibrium(self,
                         initial_coords: GTMOCoordinates,
                         max_iterations: int = 1000,
                         tolerance: float = 1e-6) -> GTMOCoordinates:
        """
        Find equilibrium point (minimum of potential energy).

        Uses gradient descent: q(n+1) = q(n) - η·∇V(q)
        """
        q = initial_coords.to_array().copy()
        learning_rate = 0.01

        for iteration in range(max_iterations):
            coords = GTMOCoordinates(*np.clip(q, 0, 1))
            grad = -self.force(coords)  # ∇V

            # Gradient descent step
            q_new = q - learning_rate * grad
            q_new = np.clip(q_new, 0, 1)

            # Check convergence
            if np.linalg.norm(q_new - q) < tolerance:
                return GTMOCoordinates(*q_new)

            q = q_new

        return GTMOCoordinates(*q)


class JuliaEmergence:
    """
    Julia set iterations for modeling semantic emergence.

    Models semantic evolution through complex dynamics:
    z(n+1) = z(n)² + c

    Adapted to GTMØ 3D space with emergence patterns.
    """

    def __init__(self, max_iterations: int = 100, escape_radius: float = 2.0):
        """
        Initialize Julia emergence model.

        Args:
            max_iterations: Maximum iteration depth
            escape_radius: Radius for escape criterion
        """
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def iterate_point(self,
                      z0: complex,
                      c: complex,
                      max_iter: Optional[int] = None) -> Tuple[int, float]:
        """
        Iterate single point in Julia set.

        Returns:
            (iterations_to_escape, final_magnitude)
        """
        if max_iter is None:
            max_iter = self.max_iterations

        z = z0
        for iteration in range(max_iter):
            z = z**2 + c

            if abs(z) > self.escape_radius:
                return iteration, abs(z)

        return max_iter, abs(z)

    def gtmo_to_complex(self, coords: GTMOCoordinates) -> complex:
        """
        Map GTMØ coordinates to complex plane.

        Uses: z = (D - 0.5) + i·(S - 0.5)
        Entropy affects iteration parameter c
        """
        real = coords.determination - 0.5
        imag = coords.stability - 0.5
        return complex(real, imag)

    def complex_to_gtmo(self, z: complex, entropy: float) -> GTMOCoordinates:
        """Map complex number back to GTMØ coordinates."""
        D = np.clip(z.real + 0.5, 0, 1)
        S = np.clip(z.imag + 0.5, 0, 1)
        E = np.clip(entropy, 0, 1)
        return GTMOCoordinates(D, S, E)

    def emergence_pattern(self, coords: GTMOCoordinates) -> Dict[str, Any]:
        """
        Analyze emergence pattern at given coordinates.

        Returns:
            - iterations: Iterations to escape
            - stability: Whether point is in Julia set
            - emergence_type: Type of semantic emergence
            - magnitude: Final iteration magnitude
        """
        z0 = self.gtmo_to_complex(coords)

        # Entropy modulates the c parameter
        c = complex(-0.4, 0.6) * (1 - coords.entropy)

        iterations, magnitude = self.iterate_point(z0, c)

        # Classify emergence pattern
        if iterations == self.max_iterations:
            emergence_type = "stable_knowledge"  # In Julia set
            is_stable = True
        elif iterations > self.max_iterations * 0.7:
            emergence_type = "slow_emergence"
            is_stable = False
        elif iterations > self.max_iterations * 0.3:
            emergence_type = "moderate_emergence"
            is_stable = False
        else:
            emergence_type = "rapid_divergence"
            is_stable = False

        return {
            'iterations': iterations,
            'is_stable': is_stable,
            'emergence_type': emergence_type,
            'magnitude': magnitude,
            'escape_velocity': magnitude / max(iterations, 1)
        }

    def compute_emergence_field(self,
                                entropy_level: float,
                                resolution: int = 50) -> np.ndarray:
        """
        Compute emergence field for given entropy level.

        Returns 2D array of iteration counts in (D, S) plane.
        """
        field = np.zeros((resolution, resolution))

        D_range = np.linspace(0, 1, resolution)
        S_range = np.linspace(0, 1, resolution)

        for i, D in enumerate(D_range):
            for j, S in enumerate(S_range):
                coords = GTMOCoordinates(D, S, entropy_level)
                pattern = self.emergence_pattern(coords)
                field[i, j] = pattern['iterations']

        return field

    def find_bifurcation_points(self,
                                coords_list: List[GTMOCoordinates]) -> List[int]:
        """
        Find bifurcation points in trajectory.

        Bifurcations occur when emergence pattern changes rapidly.
        """
        if len(coords_list) < 3:
            return []

        bifurcations = []
        patterns = [self.emergence_pattern(c) for c in coords_list]

        for i in range(1, len(patterns) - 1):
            # Check for rapid change in iterations
            delta_prev = abs(patterns[i]['iterations'] - patterns[i-1]['iterations'])
            delta_next = abs(patterns[i+1]['iterations'] - patterns[i]['iterations'])

            # Bifurcation if change exceeds threshold
            if delta_prev > 10 or delta_next > 10:
                bifurcations.append(i)

        return bifurcations


class ContextualDynamicsProcessor:
    """
    Process contextual dynamics - temporal evolution of semantic meaning.

    Combines Hamiltonian dynamics with Julia emergence to model
    how context influences semantic trajectory.
    """

    def __init__(self):
        self.hamiltonian = SemanticHamiltonian()
        self.julia = JuliaEmergence()
        self.context_memory = []

    def process_context_sequence(self,
                                  contexts: List[str],
                                  initial_coords: GTMOCoordinates,
                                  coord_calculator) -> SemanticTrajectory:
        """
        Process sequence of contexts and track semantic evolution.

        Args:
            contexts: List of text contexts
            initial_coords: Starting point
            coord_calculator: Function to calculate GTMØ coords from text

        Returns:
            SemanticTrajectory showing evolution
        """
        points = [initial_coords]
        energies = [self.hamiltonian.potential_energy(initial_coords)]
        timestamps = [0.0]

        current_coords = initial_coords
        momentum = np.zeros(3)

        for i, context in enumerate(contexts):
            # Calculate target coordinates from context
            target_coords = coord_calculator(context)

            # Compute force towards target
            direction = target_coords.to_array() - current_coords.to_array()
            force_magnitude = np.linalg.norm(direction)

            if force_magnitude > 0:
                force = direction / force_magnitude * 0.5
            else:
                force = np.zeros(3)

            # Update momentum and position
            momentum = momentum * 0.9 + force * self.hamiltonian.dt
            new_position = current_coords.to_array() + momentum * self.hamiltonian.dt
            new_position = np.clip(new_position, 0, 1)

            current_coords = GTMOCoordinates(*new_position)

            # Record trajectory
            points.append(current_coords)
            energies.append(self.hamiltonian.potential_energy(current_coords))
            timestamps.append(i + 1)

        trajectory = SemanticTrajectory(
            points=points,
            timestamps=timestamps,
            energies=energies,
            contexts=['initial'] + contexts
        )

        self.context_memory.append(trajectory)
        return trajectory

    def analyze_trajectory(self, trajectory: SemanticTrajectory) -> Dict[str, Any]:
        """
        Analyze semantic trajectory for key features.

        Returns:
            - total_distance: Total path length
            - mean_energy: Average energy
            - energy_variance: Energy fluctuation
            - stability_regions: Regions of stable evolution
            - bifurcation_points: Points of rapid change
            - emergence_types: Distribution of emergence patterns
        """
        # Basic metrics
        total_distance = trajectory.length()
        mean_energy = trajectory.mean_energy()
        energy_variance = np.var(trajectory.energies) if trajectory.energies else 0.0

        # Find stability regions (low energy variance)
        stability_regions = []
        window_size = 5
        for i in range(len(trajectory.points) - window_size):
            window_energies = trajectory.energies[i:i+window_size]
            if np.var(window_energies) < 0.01:
                stability_regions.append(i)

        # Find bifurcations using Julia analysis
        bifurcations = self.julia.find_bifurcation_points(trajectory.points)

        # Analyze emergence types along trajectory
        emergence_types = []
        for coords in trajectory.points[::5]:  # Sample every 5 points
            pattern = self.julia.emergence_pattern(coords)
            emergence_types.append(pattern['emergence_type'])

        from collections import Counter
        emergence_distribution = Counter(emergence_types)

        return {
            'total_distance': total_distance,
            'mean_energy': mean_energy,
            'energy_variance': energy_variance,
            'num_stability_regions': len(stability_regions),
            'stability_regions': stability_regions[:10],  # First 10
            'num_bifurcations': len(bifurcations),
            'bifurcation_points': bifurcations,
            'emergence_distribution': dict(emergence_distribution),
            'final_coords': trajectory.points[-1] if trajectory.points else None
        }


class DynamicsVisualizer:
    """
    Visualization tools for GTMØ dynamics.

    Provides:
    - 3D trajectory plots in GTMØ space
    - Energy evolution plots
    - Julia emergence field visualizations
    - Phase portraits
    """

    def __init__(self):
        if not HAS_PLOTTING:
            print("⚠️  Visualization not available - matplotlib required")
        self.has_plotting = HAS_PLOTTING

    def plot_trajectory_3d(self,
                          trajectory: SemanticTrajectory,
                          title: str = "Semantic Trajectory in GTMØ Space",
                          save_path: Optional[str] = None):
        """Plot 3D trajectory in (D, S, E) space."""
        if not self.has_plotting:
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract coordinates
        D = [p.determination for p in trajectory.points]
        S = [p.stability for p in trajectory.points]
        E = [p.entropy for p in trajectory.points]

        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(D)))

        # Plot trajectory
        for i in range(len(D) - 1):
            ax.plot(D[i:i+2], S[i:i+2], E[i:i+2],
                   color=colors[i], linewidth=2, alpha=0.7)

        # Mark start and end
        ax.scatter(D[0], S[0], E[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(D[-1], S[-1], E[-1], c='red', s=100, marker='*', label='End')

        # Labels
        ax.set_xlabel('Determination (D)', fontsize=12)
        ax.set_ylabel('Stability (S)', fontsize=12)
        ax.set_zlabel('Entropy (E)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Saved trajectory plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_energy_evolution(self,
                             trajectory: SemanticTrajectory,
                             title: str = "Energy Evolution",
                             save_path: Optional[str] = None):
        """Plot energy evolution over time."""
        if not self.has_plotting:
            return

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(trajectory.timestamps, trajectory.energies,
               linewidth=2, color='blue', label='Total Energy')
        ax.fill_between(trajectory.timestamps, trajectory.energies,
                        alpha=0.3, color='blue')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Energy H(q,p)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Saved energy plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_emergence_field(self,
                            julia: JuliaEmergence,
                            entropy_level: float = 0.5,
                            resolution: int = 100,
                            title: str = "Emergence Field",
                            save_path: Optional[str] = None):
        """Plot Julia emergence field at given entropy level."""
        if not self.has_plotting:
            return

        print(f"Computing emergence field at E={entropy_level:.2f}...")
        field = julia.compute_emergence_field(entropy_level, resolution)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(field.T, extent=[0, 1, 0, 1], origin='lower',
                      cmap='hot', aspect='auto')

        ax.set_xlabel('Determination (D)', fontsize=12)
        ax.set_ylabel('Stability (S)', fontsize=12)
        ax.set_title(f"{title} (E={entropy_level:.2f})",
                    fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Iterations to Escape', fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Saved emergence field to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_phase_portrait(self,
                           trajectories: List[SemanticTrajectory],
                           projection: str = 'DS',
                           title: str = "Phase Portrait",
                           save_path: Optional[str] = None):
        """
        Plot phase portrait (2D projection of trajectories).

        Args:
            trajectories: List of trajectories to plot
            projection: Which plane to project onto ('DS', 'DE', or 'SE')
            title: Plot title
            save_path: Path to save figure
        """
        if not self.has_plotting:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Select coordinates based on projection
        coord_map = {
            'DS': ('determination', 'stability', 'D', 'S'),
            'DE': ('determination', 'entropy', 'D', 'E'),
            'SE': ('stability', 'entropy', 'S', 'E')
        }

        if projection not in coord_map:
            projection = 'DS'

        attr1, attr2, label1, label2 = coord_map[projection]

        # Plot each trajectory
        for i, traj in enumerate(trajectories):
            x = [getattr(p, attr1) for p in traj.points]
            y = [getattr(p, attr2) for p in traj.points]

            colors = plt.cm.viridis(np.linspace(0, 1, len(x)))

            for j in range(len(x) - 1):
                ax.plot(x[j:j+2], y[j:j+2],
                       color=colors[j], linewidth=1.5, alpha=0.6)

            # Mark start
            ax.scatter(x[0], y[0], c='green', s=80, marker='o',
                      edgecolors='black', linewidths=1.5, zorder=10)

        ax.set_xlabel(f'{label1} Coordinate', fontsize=12)
        ax.set_ylabel(f'{label2} Coordinate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Saved phase portrait to {save_path}")
        else:
            plt.show()

        plt.close()


def demo_dynamics():
    """Demonstration of GTMØ dynamics capabilities."""
    print("\n" + "="*70)
    print("GTMØ DYNAMICS - Demonstration")
    print("="*70 + "\n")

    # 1. Hamiltonian Dynamics
    print("1️⃣  HAMILTONIAN DYNAMICS")
    print("-" * 70)

    hamiltonian = SemanticHamiltonian(mass=1.0, dt=0.01)

    # Start from high entropy state
    initial = GTMOCoordinates(determination=0.3, stability=0.2, entropy=0.8)
    print(f"   Initial state: {initial}")
    print(f"   Initial energy: {hamiltonian.potential_energy(initial):.4f}")

    # Evolve system
    print("   Evolving system for 100 steps...")
    trajectory = hamiltonian.evolve(initial, steps=100)

    print(f"   Final state: {trajectory.points[-1]}")
    print(f"   Final energy: {trajectory.energies[-1]:.4f}")
    print(f"   Trajectory length: {trajectory.length():.4f}")
    print(f"   Mean energy: {trajectory.mean_energy():.4f}")

    # Find equilibrium
    print("\n   Finding equilibrium point...")
    equilibrium = hamiltonian.find_equilibrium(initial)
    print(f"   Equilibrium: {equilibrium}")
    print(f"   Equilibrium energy: {hamiltonian.potential_energy(equilibrium):.4f}")

    # 2. Julia Emergence
    print("\n\n2️⃣  JULIA EMERGENCE ANALYSIS")
    print("-" * 70)

    julia = JuliaEmergence(max_iterations=100)

    test_points = [
        GTMOCoordinates(0.5, 0.5, 0.2),  # Balanced, low entropy
        GTMOCoordinates(0.8, 0.7, 0.3),  # High determination
        GTMOCoordinates(0.3, 0.4, 0.7),  # High entropy
    ]

    for coords in test_points:
        pattern = julia.emergence_pattern(coords)
        print(f"\n   Point: {coords}")
        print(f"   → Iterations: {pattern['iterations']}")
        print(f"   → Type: {pattern['emergence_type']}")
        print(f"   → Stable: {pattern['is_stable']}")
        print(f"   → Escape velocity: {pattern['escape_velocity']:.4f}")

    # 3. Contextual Dynamics
    print("\n\n3️⃣  CONTEXTUAL DYNAMICS")
    print("-" * 70)

    processor = ContextualDynamicsProcessor()

    # Simulated context sequence
    contexts = [
        "Nauka i wiedza",
        "Pewność i stabilność",
        "Wątpliwość i niepewność",
        "Emergence i odkrycie"
    ]

    # Simple coordinate calculator (mock)
    def mock_calculator(text: str) -> GTMOCoordinates:
        """Mock calculator for demo."""
        if "wiedza" in text.lower():
            return GTMOCoordinates(0.7, 0.6, 0.3)
        elif "stabilność" in text.lower():
            return GTMOCoordinates(0.8, 0.9, 0.2)
        elif "niepewność" in text.lower():
            return GTMOCoordinates(0.3, 0.3, 0.8)
        elif "odkrycie" in text.lower():
            return GTMOCoordinates(0.6, 0.5, 0.5)
        else:
            return GTMOCoordinates(0.5, 0.5, 0.5)

    initial = GTMOCoordinates(0.5, 0.5, 0.5)
    context_traj = processor.process_context_sequence(
        contexts, initial, mock_calculator
    )

    print(f"   Processed {len(contexts)} contexts")
    print(f"   Trajectory points: {len(context_traj.points)}")

    analysis = processor.analyze_trajectory(context_traj)
    print(f"\n   📊 Trajectory Analysis:")
    print(f"   → Total distance: {analysis['total_distance']:.4f}")
    print(f"   → Mean energy: {analysis['mean_energy']:.4f}")
    print(f"   → Energy variance: {analysis['energy_variance']:.4f}")
    print(f"   → Stability regions: {analysis['num_stability_regions']}")
    print(f"   → Bifurcation points: {analysis['num_bifurcations']}")
    print(f"   → Emergence types: {analysis['emergence_distribution']}")
    print(f"   → Final state: {analysis['final_coords']}")

    # 4. Visualization (if available)
    if HAS_PLOTTING:
        print("\n\n4️⃣  VISUALIZATION")
        print("-" * 70)

        visualizer = DynamicsVisualizer()

        print("   Generating visualizations...")
        visualizer.plot_trajectory_3d(trajectory,
                                     title="Hamiltonian Evolution",
                                     save_path="hamiltonian_trajectory.png")

        visualizer.plot_energy_evolution(trajectory,
                                        save_path="energy_evolution.png")

        visualizer.plot_emergence_field(julia, entropy_level=0.5,
                                       resolution=50,
                                       save_path="emergence_field.png")

        visualizer.plot_phase_portrait([trajectory, context_traj],
                                      projection='DS',
                                      title="Phase Portrait (D-S Plane)",
                                      save_path="phase_portrait.png")

        print("   ✅ All visualizations saved!")
    else:
        print("\n\n4️⃣  VISUALIZATION")
        print("-" * 70)
        print("   ⚠️  Matplotlib not available - skipping visualizations")

    print("\n" + "="*70)
    print("✅ GTMØ DYNAMICS DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_dynamics()
