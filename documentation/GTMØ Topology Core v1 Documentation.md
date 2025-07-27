# GTMØ Topology Core v1 Documentation

## Table of Contents
- [Overview](#overview)
- [Theory Background](#theory-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Mathematical Constants](#mathematical-constants)
- [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

GTMØ (Geometry Topology Mathematics Ø) Topology Core v1 is a Python implementation of the GTMØ theory developed by Grzegorz Skuza. This framework provides mathematical tools for analyzing linguistic subjectivity through topological methods in a three-dimensional phase space.

### Key Features
- **Phase Space Analysis**: 3D semantic space with determination, stability, and entropy dimensions
- **Topological Attractors**: Eight knowledge types that classify information naturally
- **Trajectory Analysis**: Track semantic evolution of concepts over time
- **Dynamic Modeling**: Cognitive space that "breathes" to model knowledge boundaries
- **Visualization**: 3D plotting capabilities for phase space exploration

## Theory Background

### Phase Space Dimensions

The GTMØ phase space consists of three fundamental dimensions:

1. **Determination** (X-axis, 0→1): How unambiguous or clearly defined the meaning is
   - 0 = completely unclear
   - 1 = absolutely clear

2. **Stability** (Y-axis, 0→1): How constant the meaning is over time
   - 0 = constantly changing
   - 1 = unchanging

3. **Entropy** (Z-axis, 0→1): How chaotic, creative, or paradoxical the meaning is
   - 0 = perfect order, no creativity
   - 1 = maximum chaos and creativity

### Knowledge Types (Attractors)

The system defines eight topological attractors representing different epistemic states:

| Type | Symbol | Description | Position [D,S,E] |
|------|--------|-------------|------------------|
| Singularity | Ø | Paradoxes, logical contradictions | [1.0, 1.0, 0.0] |
| Knowledge Particle | Ψᴷ | Clear, stable information | [0.85, 0.85, 0.15] |
| Knowledge Shadow | Ψʰ | Unclear, unstable information | [0.15, 0.15, 0.85] |
| Emergent | Ψᴺ | Information creating new meanings | [0.5, 0.3, 0.9] |
| Transcendent | Ψ↑ | Transcendent knowledge | [0.7, 0.7, 0.3] |
| Flux | Ψ~ | Fluid, changing knowledge | [0.5, 0.5, 0.8] |
| Void | Ψ◊ | Void or absent knowledge | [0.0, 0.0, 0.5] |
| Alienated | ℓ∅ | Alienated numbers (hybrid meanings) | [0.999, 0.999, 0.001] |

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib (optional, for visualization)

### Basic Installation
```bash
pip install numpy
pip install matplotlib  # Optional, for visualization
```

### Download
```bash
# Download the core module
wget https://example.com/gtmo_topology_core_v1.py
```

## Quick Start

```python
from gtmo_topology_core_v1 import GTMOTopologyCore, PhasePoint

# Initialize the system
gtmo = GTMOTopologyCore()

# Classify a concept
point = PhasePoint(
    determination=0.95,
    stability=0.95,
    entropy=0.05,
    label="Mathematical fact: 2+2=4"
)
knowledge_type, confidence = gtmo.classify_point(point)
print(f"Type: {knowledge_type.value}, Confidence: {confidence:.2%}")

# Visualize the phase space
gtmo.visualize_phase_space([point])
```

## API Reference

### Classes

#### `GTMOTopologyCore`
Main class implementing the GTMØ topology system.

##### Constructor
```python
GTMOTopologyCore(attractors: Optional[List[TopologicalAttractor]] = None)
```
- `attractors`: Custom list of attractors (defaults to standard GTMØ set)

##### Methods

###### `calculate_distance(point, attractor) -> float`
Calculate effective distance from a point to an attractor.
- **Parameters**:
  - `point`: `PhasePoint` or `np.ndarray` - Point in phase space
  - `attractor`: `TopologicalAttractor` - Target attractor
- **Returns**: Effective distance (weighted by attractor strength)

###### `classify_point(point) -> Tuple[KnowledgeType, float]`
Classify a point by finding the nearest attractor.
- **Parameters**:
  - `point`: `PhasePoint` or `np.ndarray` - Point to classify
- **Returns**: Tuple of (knowledge_type, confidence)

###### `analyze_trajectory(trajectory) -> Dict`
Analyze a path through phase space.
- **Parameters**:
  - `trajectory`: `List[PhasePoint]` - Sequence of points
- **Returns**: Dictionary with trajectory metrics

###### `breathe(delta_phase: float = 0.1) -> None`
Update the breathing phase of cognitive space.
- **Parameters**:
  - `delta_phase`: Phase increment in radians

###### `visualize_phase_space(points, show_attractors, show_trajectories) -> None`
Create 3D visualization of the phase space.
- **Parameters**:
  - `points`: Optional list of points to plot
  - `show_attractors`: Whether to show attractors (default: True)
  - `show_trajectories`: Connect points as trajectory (default: False)

#### `PhasePoint`
Represents a point in GTMØ phase space.

```python
@dataclass
class PhasePoint:
    determination: float  # 0-1
    stability: float      # 0-1
    entropy: float        # 0-1
    label: Optional[str] = None
```

##### Methods
- `to_array() -> np.ndarray`: Convert to numpy array
- `from_array(arr, label=None) -> PhasePoint`: Create from array

#### `TopologicalAttractor`
Represents an attractor in phase space.

```python
@dataclass
class TopologicalAttractor:
    name: str
    knowledge_type: KnowledgeType
    position: np.ndarray  # [determination, stability, entropy]
    radius: float         # Basin of attraction radius
    strength: float       # Attraction strength
```

### Enumerations

#### `KnowledgeType`
Enumeration of epistemic knowledge types.
```python
class KnowledgeType(Enum):
    PARTICLE = "Ψᴷ"
    SHADOW = "Ψʰ"
    EMERGENT = "Ψᴺ"
    LIMINALITY = "Ψˡ"
    SINGULARITY = "Ø"
    TRANSCENDENT = "Ψ↑"
    FLUX = "Ψ~"
    VOID = "Ψ◊"
    ALIENATED = "ℓ∅"
```

### Utility Functions

#### `create_semantic_trajectory(word, time_points, labels=None) -> List[PhasePoint]`
Create a semantic trajectory for a word.
- **Parameters**:
  - `word`: The word being traced
  - `time_points`: List of (determination, stability, entropy) tuples
  - `labels`: Optional labels for each time point
- **Returns**: List of PhasePoint objects

## Examples

### Example 1: Classifying Concepts

```python
from gtmo_topology_core_v1 import GTMOTopologyCore, PhasePoint

# Initialize system
gtmo = GTMOTopologyCore()

# Test various concepts
concepts = [
    PhasePoint(0.95, 0.95, 0.05, "Mathematical theorem"),
    PhasePoint(0.3, 0.4, 0.7, "Weather forecast"),
    PhasePoint(0.1, 0.1, 0.95, "Zen koan"),
    PhasePoint(0.5, 0.3, 0.9, "Internet meme")
]

for concept in concepts:
    k_type, confidence = gtmo.classify_point(concept)
    print(f"{concept.label}: {k_type.name} (confidence: {confidence:.1%})")
```

### Example 2: Tracking Word Evolution

```python
from gtmo_topology_core_v1 import create_semantic_trajectory, GTMOTopologyCore

# Create trajectory for the word "cool"
cool_trajectory = create_semantic_trajectory(
    "cool",
    [
        (0.9, 0.9, 0.1),   # 1900: Temperature description
        (0.7, 0.7, 0.3),   # 1950: Starting to mean "fashionable"
        (0.5, 0.4, 0.6),   # 1980: Multiple meanings emerge
        (0.4, 0.3, 0.7)    # 2020: Highly contextual slang
    ],
    ["1900", "1950", "1980", "2020"]
)

# Analyze the trajectory
gtmo = GTMOTopologyCore()
analysis = gtmo.analyze_trajectory(cool_trajectory)

print(f"Total semantic drift: {analysis['total_distance']:.3f}")
print(f"Dominant knowledge types: {analysis['dominant_types']}")
```

### Example 3: Dynamic Cognitive Space

```python
import numpy as np
from gtmo_topology_core_v1 import GTMOTopologyCore, PhasePoint

gtmo = GTMOTopologyCore()

# Simulate breathing cognitive space
test_point = PhasePoint(0.5, 0.5, 0.5)  # Neutral point

for i in range(10):
    # Classify before breathing
    k_type1, conf1 = gtmo.classify_point(test_point)
    
    # Breathe the space
    gtmo.breathe(np.pi / 4)
    
    # Classify after breathing
    k_type2, conf2 = gtmo.classify_point(test_point)
    
    if k_type1 != k_type2:
        print(f"Step {i}: Classification changed from {k_type1.name} to {k_type2.name}")
```

## Mathematical Constants

The system uses several fundamental constants based on GTMØ theory:

| Constant | Symbol | Value | Description |
|----------|--------|-------|-------------|
| Golden Ratio | φ | 1.618... | Natural proportion for self-organizing systems |
| Quantum Amplitude | 1/√2 | 0.707... | Maximum quantum coherence between states |
| Cognitive Center | - | [0.5, 0.5, 0.5] | Neutral knowledge state |
| Boundary Thickness | - | 0.02 | Epistemic boundary thickness |
| Entropy Threshold | - | 0.001 | Threshold for singularity collapse |
| Breathing Amplitude | - | 0.1 | Cognitive space pulsation amplitude |

## Visualization

The system includes 3D visualization capabilities using Matplotlib:

```python
import numpy as np
from gtmo_topology_core_v1 import GTMOTopologyCore, PhasePoint

# Generate sample data
gtmo = GTMOTopologyCore()
points = []

# Create clusters around different attractors
np.random.seed(42)
for attractor in gtmo.attractors[:3]:
    for _ in range(20):
        offset = np.random.randn(3) * 0.1
        pos = np.clip(attractor.position + offset, 0, 1)
        points.append(PhasePoint.from_array(pos))

# Visualize
gtmo.visualize_phase_space(
    points=points,
    show_attractors=True,
    show_trajectories=False
)
```

## Advanced Usage

### Custom Attractors

```python
from gtmo_topology_core_v1 import GTMOTopologyCore, TopologicalAttractor, KnowledgeType
import numpy as np

# Define custom attractors
custom_attractors = [
    TopologicalAttractor(
        name="Custom Knowledge",
        knowledge_type=KnowledgeType.PARTICLE,
        position=np.array([0.6, 0.6, 0.4]),
        radius=0.2,
        strength=1.5
    )
]

# Initialize with custom attractors
gtmo = GTMOTopologyCore(attractors=custom_attractors)
```

### Batch Processing

```python
def batch_classify(gtmo, texts_with_coords):
    """Classify multiple texts with their phase space coordinates."""
    results = []
    for text, (d, s, e) in texts_with_coords:
        point = PhasePoint(d, s, e, text)
        k_type, conf = gtmo.classify_point(point)
        results.append({
            'text': text,
            'type': k_type.name,
            'confidence': conf,
            'coordinates': [d, s, e]
        })
    return results
```

### Export/Import Trajectories

```python
import json

def export_trajectory(trajectory, filename):
    """Export trajectory to JSON."""
    data = []
    for point in trajectory:
        data.append({
            'determination': point.determination,
            'stability': point.stability,
            'entropy': point.entropy,
            'label': point.label
        })
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def import_trajectory(filename):
    """Import trajectory from JSON."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return [PhasePoint(**item) for item in data]
```

## Contributing

Contributions to GTMØ Topology Core are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Add unit tests for new features

## License

This implementation is based on the GTMØ theory by Grzegorz Skuza. Please refer to the original theory documentation for licensing information.

---

For more information about GTMØ theory, visit the [official repository](https://github.com/gtmo/theory) or read the [theoretical paper](https://gtmo.theory/paper.pdf).
