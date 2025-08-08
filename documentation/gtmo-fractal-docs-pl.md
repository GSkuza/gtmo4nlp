# GTMØ - Geometria Fraktalna Znaczeń

## 📊 Przegląd Systemu

**GTMØ** (Geometry Topology Mathematics Ø) to rewolucyjna teoria matematyczna modelująca subiektywność języka poprzez geometrię fraktalną. Zamiast traktować znaczenia jako punkty, GTMØ przedstawia je jako **dynamiczne trajektorie** w przestrzeni fraktalnej.

### Kluczowe Innowacje
- 🌀 **Fraktalne granice basenów** między znaczeniami
- 🔄 **Iterowane Systemy Funkcji (IFS)** dla ewolucji znaczeń
- 💫 **Zbiory Julia** w przestrzeni semantycznej
- 🌊 **Dziwne atraktory** i trajektorie chaotyczne

---

## 🧮 Podstawy Matematyczne

### Stałe Fraktalne

```python
PHI = 1.618...           # Złoty podział - samopodobieństwo
FEIGENBAUM_DELTA = 4.669 # Stała podwojenia okresu
HAUSDORFF_DIM = 1.585    # Wymiar fraktalny granic semantycznych
```

### Przestrzeń Semantyczna

Każde znaczenie opisane jest przez **trzy wymiary**:

| Wymiar | Symbol | Zakres | Opis |
|--------|--------|--------|------|
| **Determinacja** | D | [0,1] | Jak jednoznaczne jest znaczenie |
| **Stabilność** | S | [0,1] | Jak stałe w czasie |
| **Entropia** | E | [0,1] | Jak chaotyczne/kreatywne |

---

## 🎯 Atraktory Fraktalne

System definiuje **5 głównych atraktorów** z unikalnymi parametrami Julia:

### 1. **Ψᴷ - Cząstka Wiedzy**
- **Pozycja**: [0.85, 0.85, 0.15]
- **Parametr Julia**: c = -0.4 + 0.6i
- **Natura**: Stabilna, zwarta wiedza
- **Przykład**: "2+2=4"

### 2. **Ψʰ - Cień Wiedzy**
- **Pozycja**: [0.15, 0.15, 0.85]
- **Parametr Julia**: c = 0.285 + 0.01i
- **Natura**: Niepewna, rozproszona
- **Przykład**: "Może jutro będzie padać"

### 3. **Ψᴺ - Emergentna**
- **Pozycja**: [0.50, 0.30, 0.90]
- **Parametr Julia**: c = -0.835 - 0.2321i
- **Natura**: Tworząca nowe znaczenia
- **Przykład**: "Facebook", "deadline"

### 4. **Ø - Singularność**
- **Pozycja**: [1.00, 1.00, 0.00]
- **Parametr Julia**: c = -0.8 + 0.156i
- **Natura**: Paradoksy, sprzeczności
- **Przykład**: "To zdanie jest fałszywe"

### 5. **Ψ~ - Flux**
- **Pozycja**: [0.50, 0.50, 0.80]
- **Parametr Julia**: c = 0.45 + 0.1428i
- **Natura**: Płynne, zmienne znaczenia
- **Przykład**: "Cool", "vibe"

---

## 🔄 Operacje Fraktalne

### 1. Iteracja Znaczenia

```python
def iterate_meaning(point, attractor, steps):
    """
    Iteruje punkt semantyczny przez transformację fraktalną.
    
    Proces:
    1. z = punkt.to_complex()
    2. z_new = z² + c (parametr Julia atraktora)
    3. Dodaj szum semantyczny
    4. Sprawdź ucieczkę do nieskończoności
    """
```

**Przykład trajektorii słowa "consciousness":**
```
Initial: D=0.432, S=0.567, E=0.234
Iter 1:  D=0.123, S=0.789, E=0.456
Iter 2:  D=0.234, S=0.456, E=0.678
...
```

### 2. Wymiar Fraktalny

System oblicza **wymiar Hausdorffa** trajektorii używając metody box-counting:

```python
def calculate_fractal_dimension(points):
    """
    Szacuje wymiar fraktalny trajektorii.
    
    Typowe wartości:
    - Linia: 1.0
    - Granica fraktalna: ~1.585
    - Płaszczyzna: 2.0
    """
```

### 3. Dziwne Atraktory

System wykrywa chaotyczne zachowanie poprzez:
- **Wykładnik Lapunowa** > 0 (chaos)
- **Pokrycie przestrzeni fazowej** > 50%
- **Zachowanie nieperiodyczne** ale ograniczone

---

## 🎨 Wizualizacje

### 1. Mapa Basenów Fraktalnych

Pokazuje który atraktor dominuje w każdym regionie przestrzeni:

```
Kolory:
🔴 Czerwony - Ψᴷ (Wiedza)
🔵 Niebieski - Ψʰ (Cień)
🟢 Zielony - Ψᴺ (Emergent)
🟡 Żółty - Ø (Singularność)
🟣 Fioletowy - Ψ~ (Flux)
⚫ Czarny - Ucieczka
```

### 2. Zbiory Julia

Każdy atraktor generuje unikalny zbiór Julia:
- **Czarne obszary**: Punkty stabilne (pozostają w zbiorze)
- **Kolorowe gradienty**: Prędkość ucieczki
- **Granice fraktalne**: Nieskończenie złożone

### 3. IFS Sierpińskiego

System generuje fraktale semantyczne podobne do trójkąta Sierpińskiego:

```python
transformacje = [
    t1: (d, s) → (d*0.5, s*0.5)          # Skalowanie
    t2: (d, s) → (d*0.5+0.5, s*0.5)      # Przesunięcie X
    t3: (d, s) → (d*0.5+0.25, s*0.5+0.5) # Przesunięcie XY
]
```

---

## 💡 Interpretacja Wyników

### Klasyfikacja Znaczeń

| Typ Trajektorii | Interpretacja | Przykład |
|-----------------|---------------|----------|
| **Zbieżna** | Znaczenie stabilizuje się | "Matematyka" → Ψᴷ |
| **Oscylująca** | Znaczenie waha się | "Sztuka" ↔ różne interpretacje |
| **Chaotyczna** | Znaczenie nieprzewidywalne | "Meme" → chaos |
| **Uciekająca** | Znaczenie rozpada się | Neologizm → ∞ |

### Wymiary Fraktalne

```
1.0   - Znaczenie linearne (prosta trajektoria)
1.585 - Typowa granica semantyczna (fraktalna)
2.0   - Znaczenie wypełnia przestrzeń (maksymalny chaos)
```

---

## 🚀 Zastosowania

### 1. Analiza Języka Naturalnego
- Śledzenie ewolucji znaczeń w czasie
- Wykrywanie paradoksów i sprzeczności
- Identyfikacja emergentnych pojęć

### 2. Sztuczna Inteligencja
- Modelowanie nieostrości semantycznej
- Nawigacja w przestrzeni znaczeń
- Generowanie nowych konceptów

### 3. Lingwistyka Kognitywna
- Mapowanie trajektorii uczenia się
- Analiza granic między pojęciami
- Badanie chaosu semantycznego

---

## 📚 Przykłady Użycia

### Podstawowa Analiza

```python
# Inicjalizacja
fractal = FractalSemanticSpace()

# Projekcja tekstu
point = fractal.project_text("paradoks")

# Iteracja przez atraktor
trajectory = fractal.iterate_meaning(
    point, 
    fractal.attractors[3],  # Ø - Singularność
    steps=50
)

# Analiza chaosu
analysis = fractal.detect_strange_attractor(trajectory)
if analysis['is_strange']:
    print(f"Wykryto dziwny atraktor!")
    print(f"Wymiar: {analysis['dimension']:.3f}")
```

### Generowanie Fraktali IFS

```python
# Transformacje Sierpińskiego
transforms = sierpinski_semantic_transform()

# Generuj 1000 punktów
points = fractal.semantic_ifs(
    "język", 
    transforms, 
    iterations=1000
)

# Wizualizuj
plt.scatter(points[:, 0], points[:, 1])
```

---

## 🔬 Szczegóły Techniczne

### Funkcja Basenu Julia

```python
def basin_function(z, c):
    """
    Iteracja Julia dla basenu atraktora.
    
    z_n+1 = z_n² + c
    
    gdzie:
    - z: liczba zespolona (punkt)
    - c: parametr Julia (charakterystyka atraktora)
    """
    return z**2 + c
```

### Detekcja Granic

Granice między basenami są wykrywane gdy:
```python
neighbors = [basin[i-1,j], basin[i+1,j], 
             basin[i,j-1], basin[i,j+1]]
if len(set(neighbors)) > 1:
    # Punkt graniczny!
```

---

## 📈 Metryki i Wskaźniki

| Metryka | Zakres | Znaczenie |
|---------|--------|-----------|
| **Wykładnik Lapunowa** | [-∞, +∞] | >0 = chaos |
| **Pokrycie przestrzeni** | [0, 1] | >0.5 = dobra eksploracja |
| **Wymiar fraktalny** | [1, 2] | ~1.585 = granica semantyczna |
| **Prędkość ucieczki** | [0, ∞] | Stabilność znaczenia |

---

## 🎯 Kluczowe Wnioski

1. **Granice między znaczeniami są fraktalne** - mają wymiar niecałkowity (~1.585)

2. **Każdy typ wiedzy ma unikalny "odcisk palca" fraktalny** - parametr Julia

3. **Znaczenia ewoluują chaotycznie** ale w ramach atraktorów

4. **Emergencja nowych znaczeń** następuje na granicach fraktalnych

5. **Samopodobieństwo** - te same wzory powtarzają się w różnych skalach

---

## 📖 Bibliografia

- Skuza, G. (2024). *GTMØ as an attempt to capture the subjectivity of language*
- Mandelbrot, B. (1982). *The Fractal Geometry of Nature*
- Julia, G. (1918). *Mémoire sur l'itération des fonctions rationnelles*

---

*"Matematyka muzyki znaczeń, nie fizyka definicji"* - GTMØ