# GTMØ - Fractal Geometry of Meanings

## 📊 System Overview

**GTMØ** (Geometry Topology Mathematics Ø) is a revolutionary mathematical theory that models linguistic subjectivity through fractal geometry. Rather than treating meanings as static points, GTMØ represents them as **dynamic trajectories** in fractal space.

### Key Innovations
- 🌀 **Fractal basin boundaries** between meanings
- 🔄 **Iterated Function Systems (IFS)** for meaning evolution
- 💫 **Julia sets** in semantic space
- 🌊 **Strange attractors** and chaotic trajectories

---

## 🧮 Mathematical Foundations

### Fractal Constants

```python
PHI = 1.618...           # Golden ratio - self-similarity
FEIGENBAUM_DELTA = 4.669 # Period-doubling constant
HAUSDORFF_DIM = 1.585    # Fractal dimension of semantic boundaries
```

### Semantic Space

Each meaning is described through **three dimensions**:

| Dimension | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| **Determination** | D | [0,1] | How unambiguous the meaning is |
| **Stability** | S | [0,1] | How constant over time |
| **Entropy** | E | [0,1] | How chaotic/creative |

---

## 🎯 Fractal Attractors

The system defines **5 primary attractors** with unique Julia parameters:

### 1. **Ψᴷ - Knowledge Particle**
- **Position**: [0.85, 0.85, 0.15]
- **Julia Parameter**: c = -0.4 + 0.6i
- **Nature**: Stable, compact knowledge
- **Example**: "2+2=4"

### 2. **Ψʰ - Knowledge Shadow**
- **Position**: [0.15, 0.15, 0.85]
- **Julia Parameter**: c = 0.285 + 0.01i
- **Nature**: Uncertain, dispersed
- **Example**: "It might rain tomorrow"

### 3. **Ψᴺ - Emergent**
- **Position**: [0.50, 0.30, 0.90]
- **Julia Parameter**: c = -0.835 - 0.2321i
- **Nature**: Creating new meanings
- **Example**: "Facebook", "deadline"

### 4. **Ø - Singularity**
- **Position**: [1.00, 1.00, 0.00]
- **Julia Parameter**: c = -0.8 + 0.156i
- **Nature**: Paradoxes, contradictions
- **Example**: "This sentence is false"

### 5. **Ψ~ - Flux**
- **Position**: [0.50, 0.50, 0.80]
- **Julia Parameter**: c = 0.45 + 0.1428i
- **Nature**: Fluid, changing meanings
- **Example**: "Cool", "vibe"

---

## 🔄 Fractal Operations

### 1. Meaning Iteration

```python
def iterate_meaning(point, attractor, steps):
    """
    Iterates a semantic point through fractal transformation.
    
    Process:
    1. z = point.to_complex()
    2. z_new = z² + c (attractor's Julia parameter)
    3. Add semantic noise
    4. Check for escape to infinity
    """
```

**Example trajectory for "consciousness":**
```
Initial: D=0.432, S=0.567, E=0.234
Iter 1:  D=0.123, S=0.789, E=0.456
Iter 2:  D=0.234, S=0.456, E=0.678
...
```

### 2. Fractal Dimension

The system calculates the **Hausdorff dimension** of trajectories using the box-counting method:

```python
def calculate_fractal_dimension(points):
    """
    Estimates fractal dimension of trajectory.
    
    Typical values:
    - Line: 1.0
    - Fractal boundary: ~1.585
    - Plane: 2.0
    """
```

### 3. Strange Attractors

The system detects chaotic behaviour through:
- **Lyapunov exponent** > 0 (chaos)
- **Phase space coverage** > 50%
- **Non-periodic** yet bounded behaviour

---

## 🎨 Visualisations

### 1. Fractal Basin Map

Shows which attractor dominates each region of space:

```
Colours:
🔴 Red - Ψᴷ (Knowledge)
🔵 Blue - Ψʰ (Shadow)
🟢 Green - Ψᴺ (Emergent)
🟡 Yellow - Ø (Singularity)
🟣 Purple - Ψ~ (Flux)
⚫ Black - Escape
```

### 2. Julia Sets

Each attractor generates a unique Julia set:
- **Black regions**: Stable points (remain in set)
- **Colour gradients**: Escape velocity
- **Fractal boundaries**: Infinitely complex

### 3. Sierpinski IFS

The system generates semantic fractals similar to Sierpinski's triangle:

```python
transformations = [
    t1: (d, s) → (d*0.5, s*0.5)          # Scaling
    t2: (d, s) → (d*0.5+0.5, s*0.5)      # X-shift
    t3: (d, s) → (d*0.5+0.25, s*0.5+0.5) # XY-shift
]
```

---

## 💡 Result Interpretation

### Meaning Classification

| Trajectory Type | Interpretation | Example |
|-----------------|----------------|---------|
| **Convergent** | Meaning stabilises | "Mathematics" → Ψᴷ |
| **Oscillating** | Meaning fluctuates | "Art" ↔ various interpretations |
| **Chaotic** | Meaning unpredictable | "Meme" → chaos |
| **Escaping** | Meaning dissolves | Neologism → ∞ |

### Fractal Dimensions

```
1.0   - Linear meaning (straight trajectory)
1.585 - Typical semantic boundary (fractal)
2.0   - Meaning fills space (maximum chaos)
```

---

## 🚀 Applications

### 1. Natural Language Analysis
- Tracking meaning evolution over time
- Detecting paradoxes and contradictions
- Identifying emergent concepts

### 2. Artificial Intelligence
- Modelling semantic fuzziness
- Navigating meaning space
- Generating new concepts

### 3. Cognitive Linguistics
- Mapping learning trajectories
- Analysing boundaries between concepts
- Studying semantic chaos

---

## 📚 Usage Examples

### Basic Analysis

```python
# Initialisation
fractal = FractalSemanticSpace()

# Text projection
point = fractal.project_text("paradox")

# Iteration through attractor
trajectory = fractal.iterate_meaning(
    point, 
    fractal.attractors[3],  # Ø - Singularity
    steps=50
)

# Chaos analysis
analysis = fractal.detect_strange_attractor(trajectory)
if analysis['is_strange']:
    print(f"Strange attractor detected!")
    print(f"Dimension: {analysis['dimension']:.3f}")
```

### Generating IFS Fractals

```python
# Sierpinski transformations
transforms = sierpinski_semantic_transform()

# Generate 1000 points
points = fractal.semantic_ifs(
    "language", 
    transforms, 
    iterations=1000
)

# Visualise
plt.scatter(points[:, 0], points[:, 1])
```

---

## 🔬 Technical Details

### Julia Basin Function

```python
def basin_function(z, c):
    """
    Julia iteration for attractor basin.
    
    z_n+1 = z_n² + c
    
    where:
    - z: complex number (point)
    - c: Julia parameter (attractor characteristic)
    """
    return z**2 + c
```

### Boundary Detection

Boundaries between basins are detected when:
```python
neighbors = [basin[i-1,j], basin[i+1,j], 
             basin[i,j-1], basin[i,j+1]]
if len(set(neighbors)) > 1:
    # Boundary point!
```

---

## 📈 Metrics and Indicators

| Metric | Range | Meaning |
|--------|-------|---------|
| **Lyapunov Exponent** | [-∞, +∞] | >0 = chaos |
| **Space Coverage** | [0, 1] | >0.5 = good exploration |
| **Fractal Dimension** | [1, 2] | ~1.585 = semantic boundary |
| **Escape Velocity** | [0, ∞] | Meaning stability |

---

## 🎯 Key Insights

1. **Boundaries between meanings are fractal** - they have non-integer dimension (~1.585)

2. **Each knowledge type has a unique fractal "fingerprint"** - Julia parameter

3. **Meanings evolve chaotically** but within attractor basins

4. **New meaning emergence** occurs at fractal boundaries

5. **Self-similarity** - the same patterns repeat at different scales

---

## 📖 References

- Skuza, G. (2024). *GTMØ as an attempt to capture the subjectivity of language*
- Mandelbrot, B. (1982). *The Fractal Geometry of Nature*
- Julia, G. (1918). *Mémoire sur l'itération des fonctions rationnelles*

---

*"Mathematics of the music of meanings, not physics of definitions"* - GTMØ