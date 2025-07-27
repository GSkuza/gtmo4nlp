"""
GTMO CORE v1 - SYNTHESIS OF CONFIGURATIONAL AND ADAPTIVE APPROACHES
========================================================================

This version combines:
- Configurational Framework's AlienatedNumbers and emergence theory
- Observation lenses and adaptive metrics
- New unified theory of cognitive trajectories

Key Synthesis:
1. AlienatedNumbers emerge through specific observation lenses
2. Configuration space has learnable physics
3. Observers develop personal "ways of seeing" (lens preferences)
4. Trajectories can be observed through different lenses

Author: GTMO Research Team
Version: 1.0 (Unified Framework)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from abc import ABC, abstractmethod


# === CORE CONFIGURATIONAL COMPONENTS ===

@dataclass
class Configuration:
    """Unified configuration with automatic normalization."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0
    orientation: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    scale: float = 1.0
    
    def __post_init__(self):
        """Ensure mathematical consistency."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        
        # Normalize orientation
        norm = np.linalg.norm(self.orientation)
        if norm > 1e-9:
            self.orientation = self.orientation / norm
        else:
            self.orientation = np.array([1., 0., 0.])


@dataclass
class AlienatedNumber:
    """Mathematical object with indefinite aspects captured through configuration."""
    definable: str  # The mathematical part
    configuration: Configuration  # The indefinite part
    emergence_strength: float = 0.0
    lens_signature: Optional[str] = None  # Which lens revealed this
    
    def __str__(self):
        # Convert orientation to readable format
        if abs(np.dot(self.configuration.orientation, [1, 0, 0])) > 0.9:
            orient = "horizontal"
        elif abs(np.dot(self.configuration.orientation, [0, 1, 0])) > 0.9:
            orient = "vertical"
        else:
            orient = f"{np.degrees(np.arctan2(self.configuration.orientation[1], 
                                             self.configuration.orientation[0])):.0f}°"
        
        lens_info = f"[{self.lens_signature}]" if self.lens_signature else ""
        return f"{self.definable}_{orient}{lens_info}"
    
    def collapse(self):
        """When subjected to pure math operations, collapses to Ø."""
        return "Ø"


# === OBSERVATION LENSES (from v7.0 with extensions) ===

class ObservationLens(ABC):
    """Abstract base for observation lenses."""
    
    @abstractmethod
    def observe(self, obj1: Tuple[Any, Configuration], 
                obj2: Tuple[Any, Configuration],
                observer_config: Configuration,
                distance: float) -> Tuple[Any, Dict[str, Any]]:
        """Return (observed_value, metadata)"""
        pass
    
    @abstractmethod
    def can_reveal_alienation(self, distance: float) -> bool:
        """Can this lens reveal AlienatedNumbers at given distance?"""
        pass


class ConcatenationLens(ObservationLens):
    """Sees objects as concatenated when close."""
    
    def observe(self, obj1, obj2, observer_config, distance):
        if distance < 0.5:
            value = f"{obj1[0]}{obj2[0]}"
            return value, {"type": "concatenated"}
        else:
            value = f"{obj1[0]} + {obj2[0]}"
            return value, {"type": "separate"}
    
    def can_reveal_alienation(self, distance):
        return distance < 0.1


class CausalLens(ObservationLens):
    """Interprets spatial relationships as causal relationships."""
    
    def observe(self, obj1, obj2, observer_config, distance):
        direction = obj2[1].position - obj1[1].position
        if np.linalg.norm(direction) > 1e-9:
            direction = direction / np.linalg.norm(direction)
            
        alignment = np.dot(observer_config.orientation, direction)
        
        if alignment > 0.7:
            value = f"{obj1[0]} → {obj2[0]}"
            meta = {"type": "causal", "direction": "forward"}
        elif alignment < -0.7:
            value = f"{obj2[0]} → {obj1[0]}"
            meta = {"type": "causal", "direction": "backward"}
        else:
            value = f"{obj1[0]} || {obj2[0]}"
            meta = {"type": "parallel"}
            
        return value, meta
    
    def can_reveal_alienation(self, distance):
        return distance < 0.2  # Causal collapse at very close range


class TopologicalLens(ObservationLens):
    """Sees objects through their topological relationships."""
    
    def observe(self, obj1, obj2, observer_config, distance):
        # Check if objects form a closed loop in configuration space
        loop_strength = self._calculate_loop_strength(obj1[1], obj2[1], observer_config)
        
        if loop_strength > 0.8:
            value = f"◯({obj1[0]},{obj2[0]})"
            meta = {"type": "topological_loop", "strength": loop_strength}
        elif distance < 1.0:
            value = f"{obj1[0]}~{obj2[0]}"
            meta = {"type": "topological_neighbor"}
        else:
            value = f"{obj1[0]} ⊥ {obj2[0]}"
            meta = {"type": "topological_separate"}
            
        return value, meta
    
    def can_reveal_alienation(self, distance):
        return distance < 0.05  # Topology collapses at extreme proximity
    
    def _calculate_loop_strength(self, config1, config2, observer):
        # Simplified loop detection based on orientation alignment
        cross = np.cross(config1.orientation, config2.orientation)
        return abs(np.dot(cross, observer.orientation))


class QuantumLens(ObservationLens):
    """Observes superposition states."""
    
    def observe(self, obj1, obj2, observer_config, distance):
        if 0.3 < distance < 0.7:
            # Superposition zone
            phase = np.dot(obj1[1].orientation, obj2[1].orientation)
            value = f"|{obj1[0]}⟩ + e^(i{phase:.2f})|{obj2[0]}⟩"
            meta = {"type": "superposition", "phase": phase}
        else:
            # Collapsed states
            value = f"|{obj1[0]}{obj2[0]}⟩" if distance < 0.3 else f"|{obj1[0]}⟩|{obj2[0]}⟩"
            meta = {"type": "collapsed"}
            
        return value, meta
    
    def can_reveal_alienation(self, distance):
        return distance < 0.15


# === ADAPTIVE OBSERVER WITH LENS SYSTEM ===

class AdaptiveConfigurationalObserver:
    """
    Observer that combines:
    - Adaptive distance metrics (from v7.0)
    - Multiple observation lenses
    - AlienatedNumber detection
    - Trajectory learning
    """
    
    def __init__(self, observer_id: str, initial_config: Optional[Configuration] = None):
        self.id = observer_id
        self.configuration = initial_config or Configuration()
        
        # Adaptive distance weights
        self.distance_weights = {
            'spatial': 1.0,
            'temporal': 0.5,
            'orientation': 0.3,
            'scale': 0.2
        }
        
        # Lens system
        self.lenses = {
            'concatenation': ConcatenationLens(),
            'causal': CausalLens(),
            'topological': TopologicalLens(),
            'quantum': QuantumLens()
        }
        self.lens_performance = {name: 0.5 for name in self.lenses}
        self.current_lens = 'concatenation'
        
        # Learning parameters
        self.base_learning_rate = 0.1
        self.current_learning_rate = self.base_learning_rate
        self.error_history = []
        
        # Trajectory and pattern memory
        self.observation_history: List[Dict[str, Any]] = []
        self.alienation_events: List[AlienatedNumber] = []
        self.trajectory_patterns: Dict[str, List] = {}
        
    def calculate_distance(self, config1: Configuration, config2: Configuration) -> float:
        """Calculate weighted configuration distance."""
        components = {
            'spatial': np.linalg.norm(config1.position - config2.position),
            'temporal': abs(config1.time - config2.time),
            'orientation': 1 - abs(np.dot(config1.orientation, config2.orientation)),
            'scale': abs(np.log(config1.scale / (config2.scale + 1e-9) + 1e-9))
        }
        
        weighted_sum = sum(
            (components[key] * self.distance_weights[key])**2 
            for key in components
        )
        
        return np.sqrt(weighted_sum)
    
    def observe(self, obj1: Tuple[Any, Configuration], 
                obj2: Tuple[Any, Configuration]) -> Dict[str, Any]:
        """
        Observe two objects through current lens, potentially revealing AlienatedNumbers.
        """
        # Calculate configuration distance
        distance = self.calculate_distance(obj1[1], obj2[1])
        
        # Get observation through current lens
        lens = self.lenses[self.current_lens]
        observed_value, metadata = lens.observe(obj1, obj2, self.configuration, distance)
        
        # Check for AlienatedNumber emergence
        alienated = None
        if lens.can_reveal_alienation(distance) and distance < 0.1:
            alienated = self._create_alienated_number(
                obj1, obj2, distance, lens_name=self.current_lens
            )
            self.alienation_events.append(alienated)
            observed_value = str(alienated)
            metadata['alienated'] = True
        
        # Calculate observation confidence
        observer_distance = self.calculate_distance(
            self.configuration,
            Configuration(position=(obj1[1].position + obj2[1].position) / 2)
        )
        confidence = math.exp(-observer_distance * 0.2)
        
        # Record observation
        observation = {
            'objects': [obj1[0], obj2[0]],
            'distance': distance,
            'observed_value': observed_value,
            'lens': self.current_lens,
            'confidence': confidence,
            'metadata': metadata,
            'alienated': alienated,
            'timestamp': len(self.observation_history)
        }
        
        self.observation_history.append(observation)
        self._detect_patterns()
        
        return observation
    
    def _create_alienated_number(self, obj1, obj2, distance, lens_name):
        """Create AlienatedNumber when configuration conditions are met."""
        # Interpolate configuration at limit point
        t = 0.5  # Midpoint
        merged_config = Configuration(
            position=obj1[1].position * (1-t) + obj2[1].position * t,
            time=obj1[1].time * (1-t) + obj2[1].time * t,
            orientation=obj1[1].orientation * (1-t) + obj2[1].orientation * t,
            scale=obj1[1].scale * (1-t) + obj2[1].scale * t
        )
        
        return AlienatedNumber(
            definable=f"{obj1[0]}{obj2[0]}",
            configuration=merged_config,
            emergence_strength=math.exp(-distance / 0.1),
            lens_signature=lens_name
        )
    
    def _detect_patterns(self):
        """Detect emergence patterns in observation history."""
        if len(self.observation_history) < 3:
            return
            
        recent = self.observation_history[-3:]
        
        # Pattern 1: Distance convergence to alienation
        distances = [obs['distance'] for obs in recent]
        if all(d1 > d2 for d1, d2 in zip(distances[:-1], distances[1:])):
            if distances[-1] < 0.2 and not recent[-1].get('alienated'):
                print(f"[{self.id}] Pattern detected: Approaching alienation threshold")
        
        # Pattern 2: Lens revealing different aspects
        if len(set(obs['lens'] for obs in recent)) == len(recent):
            print(f"[{self.id}] Pattern detected: Multi-lens exploration")
    
    def switch_lens(self, new_lens: str):
        """Switch to a different observation lens."""
        if new_lens in self.lenses:
            self.current_lens = new_lens
            print(f"[{self.id}] Switched to {new_lens} lens")
    
    def learn_from_feedback(self, expected_value: Any, last_observation: Dict[str, Any]):
        """
        Learn from observation error:
        - Adjust distance weights
        - Update lens performance
        - Adapt learning rate
        """
        error = 0.0 if str(expected_value) == str(last_observation['observed_value']) else 1.0
        self.error_history.append(error)
        
        if error > 0:
            # Update lens performance
            self.lens_performance[self.current_lens] *= 0.9
            
            # Try to find better lens
            best_lens = max(self.lens_performance.items(), key=lambda x: x[1])
            if best_lens[0] != self.current_lens and best_lens[1] > self.lens_performance[self.current_lens]:
                self.switch_lens(best_lens[0])
            
            # Adapt distance weights based on error type
            if "alienated" in last_observation['metadata'] and "alienated" not in str(expected_value):
                # We saw alienation when we shouldn't have
                self.distance_weights['spatial'] *= 1.1
            elif "alienated" not in last_observation['metadata'] and "alienated" in str(expected_value):
                # We missed alienation
                self.distance_weights['spatial'] *= 0.9
        else:
            # Reinforce successful lens
            self.lens_performance[self.current_lens] = min(1.0, self.lens_performance[self.current_lens] * 1.1)
        
        # Meta-learning: adapt learning rate
        if len(self.error_history) > 10:
            recent_error = np.mean(self.error_history[-10:])
            self.current_learning_rate = self.base_learning_rate * (0.5 + recent_error)
            self.current_learning_rate = np.clip(self.current_learning_rate, 0.01, 0.5)


# === DEMONSTRATION SYSTEM ===

class UnifiedGTMOSystem:
    """System demonstrating unified approach."""
    
    def __init__(self):
        self.observers: Dict[str, AdaptiveConfigurationalObserver] = {}
        self.objects: Dict[str, Tuple[Any, Configuration]] = {}
    
    def demonstrate_lens_based_alienation(self):
        """Show how different lenses reveal different AlienatedNumbers."""
        print("=" * 70)
        print("LENS-BASED ALIENATION DEMONSTRATION")
        print("=" * 70)
        
        # Create observer
        observer = AdaptiveConfigurationalObserver("lens_explorer")
        
        # Create two objects very close together
        config1 = Configuration(position=np.array([0, 0, 0]))
        config2 = Configuration(position=np.array([0.05, 0, 0]))
        
        print("\n--- Same objects observed through different lenses ---\n")
        
        for lens_name in ['concatenation', 'causal', 'topological', 'quantum']:
            observer.switch_lens(lens_name)
            result = observer.observe(("0", config1), ("1", config2))
            
            print(f"{lens_name:15s} lens: {result['observed_value']}")
            if result.get('alienated'):
                print(f"  └─ AlienatedNumber emerged with strength {result['alienated'].emergence_strength:.3f}")
    
    def demonstrate_adaptive_learning(self):
        """Show system learning optimal configuration for observation."""
        print("\n\n" + "=" * 70)
        print("ADAPTIVE LEARNING DEMONSTRATION")
        print("=" * 70)
        
        learner = AdaptiveConfigurationalObserver("adaptive_learner")
        
        # Task: learn to see "01" as causal relationship "0 → 1"
        config1 = Configuration(position=np.array([-1, 0, 0]))
        config2 = Configuration(position=np.array([1, 0, 0]))
        
        expected = "0 → 1"
        
        print("\n--- Learning to see causal relationships ---\n")
        
        for i in range(10):
            result = learner.observe(("0", config1), ("1", config2))
            
            if i % 3 == 0:
                print(f"Step {i}: Observed '{result['observed_value']}' using {result['lens']} lens")
            
            if result['observed_value'] != expected:
                learner.learn_from_feedback(expected, result)
            else:
                print(f"\nSuccess! Learned to see '{expected}' using {result['lens']} lens")
                print(f"Final distance weights: {', '.join(f'{k}:{v:.2f}' for k,v in learner.distance_weights.items())}")
                break
    
    def demonstrate_trajectory_alienation(self):
        """Show AlienatedNumber emergence along trajectories."""
        print("\n\n" + "=" * 70)
        print("TRAJECTORY-BASED ALIENATION DEMONSTRATION")
        print("=" * 70)
        
        observer = AdaptiveConfigurationalObserver("trajectory_observer")
        
        print("\n--- Spiral approach trajectory ---\n")
        
        # Fixed first object
        obj1 = ("X", Configuration(position=np.array([0, 0, 0])))
        
        # Spiral trajectory for second object
        for t in np.linspace(0, 4*np.pi, 30):
            radius = 2 * math.exp(-t/8)
            config2 = Configuration(
                position=np.array([radius * np.cos(t), radius * np.sin(t), 0]),
                orientation=np.array([np.cos(t), np.sin(t), 0])
            )
            
            result = observer.observe(obj1, ("Y", config2))
            
            if result.get('alienated') or t == 0:
                print(f"t={t:.2f}: distance={result['distance']:.3f} → {result['observed_value']}")
        
        print(f"\nTotal AlienatedNumbers discovered: {len(observer.alienation_events)}")
        if observer.alienation_events:
            print("First alienation:", observer.alienation_events[0])
            print("Last alienation:", observer.alienation_events[-1])


def main():
    """Run unified demonstration."""
    system = UnifiedGTMOSystem()
    
    system.demonstrate_lens_based_alienation()
    system.demonstrate_adaptive_learning()
    system.demonstrate_trajectory_alienation()
    
    print("\n\n" + "=" * 70)
    print("UNIFIED FRAMEWORK SUMMARY")
    print("=" * 70)
    print("""
Key Synthesis Achieved:
1. Observation Lenses: Different ways of seeing reveal different aspects
2. AlienatedNumbers: Emerge naturally when configuration distance → 0
3. Adaptive Metrics: System learns which dimensions matter
4. Pattern Detection: Trajectories reveal emergence patterns
5. Multi-perspective: Same configuration yields different truths through different lenses

This unified approach provides:
- Theoretical rigor (AlienatedNumbers, configuration space)
- Practical adaptability (learnable metrics, switchable lenses)
- Emergent complexity (patterns arise from simple rules)
- Cognitive modeling (observers develop "ways of seeing")
    """)


if __name__ == "__main__":
    main()
