# GTMØ Polish Language Processing - Instrukcja Użycia

## 📋 Spis Treści

1. [Przegląd Modułów](#przegląd-modułów)
2. [Instalacja i Wymagania](#instalacja-i-wymagania)
3. [Problem z Spacjami w Nazwach Plików](#problem-z-spacjami-w-nazwach-plików)
4. [Jak Uruchamiać Skrypty](#jak-uruchamiać-skrypty)
5. [Przykłady Użycia](#przykłady-użycia)
6. [Testy](#testy)
7. [Troubleshooting](#troubleshooting)

---

## 📦 Przegląd Modułów

### Główne Moduły

| Plik | Opis | Rozmiar |
|------|------|---------|
| `GTMØ Polish Morphological Analysis Module.py` | Główny moduł analizy morfologicznej polszczyzny | 82 KB |
| `GTMØ Syntactic Analysis Engine.py` | Silnik analizy składniowej | 29 KB |
| `gtmo_dynamics.py` | Moduł dynamiki semantycznej (Hamiltonian, Julia) | 28 KB |
| `gtmo_processor_import.py` | **Helper do importowania** (użyj tego!) | 4 KB |

### Główne Klasy

- **GTMOProcessor** - przetwarzanie morfologiczne tekstów polskich
- **GTMOAnalyzer** - analiza składniowa
- **SemanticHamiltonian** - dynamika hamiltonowska
- **JuliaEmergence** - analiza emergence przez zbiory Julii
- **ContextualDynamicsProcessor** - dynamika kontekstowa

---

## ⚙️ Instalacja i Wymagania

### Wymagania Systemowe

```bash
Python >= 3.8
numpy >= 1.20.0
```

### Opcjonalne (dla pełnej funkcjonalności)

```bash
# Morfologia polska
morfeusz2          # Analiza morfologiczna
spacy >= 3.0       # NLP (opcjonalne)
stanza             # NLP (opcjonalne)

# Wizualizacje
matplotlib >= 3.3  # Wykresy dynamiki
plotly >= 5.0      # Interaktywne wizualizacje
```

### Instalacja Zależności

```bash
# Podstawowe
pip install numpy

# Opcjonalne - morfologia
pip install morfeusz2
pip install spacy
python -m spacy download pl_core_news_sm

# Opcjonalne - wizualizacje
pip install matplotlib plotly kaleido
```

---

## ⚠️ Problem z Spacjami w Nazwach Plików

### Dlaczego to Problem?

Pliki z spacjami w nazwach **NIE MOGĄ** być importowane w Pythonie standardowo:

```python
# ❌ TO NIE DZIAŁA!
from Polish_Language_Processing.GTMØ Polish Morphological Analysis Module import GTMOProcessor
# SyntaxError: invalid syntax
```

### Rozwiązanie: Użyj `gtmo_processor_import.py`

Stworzyliśmy pomocniczy moduł, który rozwiązuje ten problem używając `importlib`.

---

## 🚀 Jak Uruchamiać Skrypty

### Metoda 1: Użyj Helpera `gtmo_processor_import.py` (ZALECANE ✅)

To **najprostszy i najlepszy** sposób.

#### Przykład 1: GTMOProcessor (Morfologia)

```python
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor

# Utwórz procesor
processor = get_gtmo_processor()

# Analizuj tekst
text = "Einstein udowodnił, że czas jest względny."
coords, config, metadata = processor.calculate_coordinates(text)

# Wyświetl wyniki
print(f"Determination: {coords.determination:.3f}")
print(f"Stability:     {coords.stability:.3f}")
print(f"Entropy:       {coords.entropy:.3f}")
```

#### Przykład 2: GTMOAnalyzer (Składnia)

```python
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_analyzer

# Utwórz analyzer
analyzer = get_gtmo_analyzer()

# Analizuj składnię (jeśli zaimplementowane)
# analysis = analyzer.analyze(text)
```

#### Przykład 3: Dynamika GTMØ

```python
from Polish_Language_Processing.gtmo_processor_import import (
    get_hamiltonian,
    get_julia_analyzer,
    get_contextual_processor
)

# Hamiltonian
hamiltonian = get_hamiltonian()

# Julia emergence
julia = get_julia_analyzer()

# Contextual dynamics
contextual = get_contextual_processor()
```

#### Przykład 4: Import Wszystkich Klas Dynamiki

```python
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_dynamics

# Pobierz wszystkie klasy
dynamics = get_gtmo_dynamics()

# Użyj klas
SemanticHamiltonian = dynamics['SemanticHamiltonian']
JuliaEmergence = dynamics['JuliaEmergence']
GTMOCoordinates = dynamics['GTMOCoordinates']

# Utwórz instancje
hamiltonian = SemanticHamiltonian()
julia = JuliaEmergence()
coords = GTMOCoordinates(0.5, 0.5, 0.5)
```

---

### Metoda 2: Bezpośrednie Uruchomienie (dla modułów bez spacji)

#### gtmo_dynamics.py

```bash
# Uruchom demo wbudowane w moduł
cd Polish_Language_Processing
python gtmo_dynamics.py
```

lub z poziomu głównego katalogu:

```python
# W skrypcie Python
import sys
sys.path.insert(0, 'Polish_Language_Processing')
import gtmo_dynamics

# Użyj klas
hamiltonian = gtmo_dynamics.SemanticHamiltonian()
```

---

### Metoda 3: Użyj `importlib` (zaawansowane)

Jeśli musisz zaimportować plik ze spacjami bezpośrednio:

```python
import importlib.util
import sys
import os

# Ścieżka do pliku
module_path = os.path.join(
    'Polish_Language_Processing',
    'GTMØ Polish Morphological Analysis Module.py'
)

# Załaduj moduł
spec = importlib.util.spec_from_file_location("gtmo_morphology", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules['gtmo_morphology'] = module
spec.loader.exec_module(module)

# Użyj klasy
processor = module.GTMOProcessor()
```

---

### Metoda 4: Uruchom Demo Skrypty

#### Demo Morfologii Rozszerzonej

```bash
# Z katalogu głównego projektu
python demo_extended_features.py
```

**Co pokazuje:**
- Nieregularne czasowniki
- Analiza derywacyjna (prefiksy/sufiksy)
- Kolokacje
- Embeddings semantyczne

#### Demo Dynamiki GTMØ

```bash
python demo_gtmo_dynamics.py
```

**Co pokazuje:**
1. Ewolucja hamiltonowska
2. Zbiory Julii i emergence
3. Dynamika kontekstowa
4. Analiza prawdziwych tekstów polskich
5. Wizualizacje (jeśli matplotlib dostępny)

---

## 💡 Przykłady Użycia

### Przykład 1: Prosta Analiza Tekstu

```python
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor

processor = get_gtmo_processor()

# Analizuj zdanie
text = "Kot śpi na macie."
coords, config, metadata = processor.calculate_coordinates(text)

print("=== WYNIKI ANALIZY ===")
print(f"Tekst: {text}")
print(f"D (Określoność):  {coords.determination:.3f}")
print(f"S (Stabilność):   {coords.stability:.3f}")
print(f"E (Entropia):     {coords.entropy:.3f}")
print(f"\nKonfiguracja: {config}")
print(f"Metadata: {metadata}")
```

---

### Przykład 2: Ewolucja Hamiltonowska

```python
from Polish_Language_Processing.gtmo_processor_import import get_hamiltonian
from Polish_Language_Processing.gtmo_dynamics import GTMOCoordinates

# Utwórz hamiltonian
hamiltonian = get_hamiltonian()

# Punkt startowy - stan nieokreślony
initial = GTMOCoordinates(
    determination=0.3,
    stability=0.2,
    entropy=0.8
)

print(f"Początek: {initial}")
print(f"Energia początkowa: {hamiltonian.potential_energy(initial):.4f}")

# Ewoluuj system
trajectory = hamiltonian.evolve(initial, steps=100)

print(f"\nKoniec: {trajectory.points[-1]}")
print(f"Energia końcowa: {trajectory.energies[-1]:.4f}")
print(f"Długość trajektorii: {trajectory.length():.4f}")
```

---

### Przykład 3: Analiza Emergence

```python
from Polish_Language_Processing.gtmo_processor_import import get_julia_analyzer
from Polish_Language_Processing.gtmo_dynamics import GTMOCoordinates

julia = get_julia_analyzer()

# Testowe punkty
points = {
    "Wiedza stabilna": GTMOCoordinates(0.85, 0.80, 0.15),
    "Stan chaotyczny": GTMOCoordinates(0.15, 0.10, 0.95),
}

for label, coords in points.items():
    pattern = julia.emergence_pattern(coords)

    print(f"\n{label}: {coords}")
    print(f"  Typ emergence: {pattern['emergence_type']}")
    print(f"  Stabilny: {'TAK' if pattern['is_stable'] else 'NIE'}")
    print(f"  Iteracje: {pattern['iterations']}")
```

---

### Przykład 4: Dynamika Kontekstowa

```python
from Polish_Language_Processing.gtmo_processor_import import (
    get_gtmo_processor,
    get_contextual_processor
)
from Polish_Language_Processing.gtmo_dynamics import GTMOCoordinates

# Procesory
processor = get_gtmo_processor()
contextual = get_contextual_processor()

# Funkcja do obliczania współrzędnych z tekstu
def text_to_coords(text):
    coords, _, _ = processor.calculate_coordinates(text)
    return GTMOCoordinates(
        coords.determination,
        coords.stability,
        coords.entropy
    )

# Sekwencja kontekstów
contexts = [
    "Nauka i wiedza są fundamentem cywilizacji.",
    "Wątpliwości prowadzą do głębszej refleksji.",
    "Emergence nowej wiedzy wymaga odwagi."
]

# Punkt startowy
initial = GTMOCoordinates(0.5, 0.5, 0.5)

# Przetwórz sekwencję
trajectory = contextual.process_context_sequence(
    contexts, initial, text_to_coords
)

# Analiza
analysis = contextual.analyze_trajectory(trajectory)

print(f"Całkowita odległość: {analysis['total_distance']:.4f}")
print(f"Średnia energia: {analysis['mean_energy']:.4f}")
print(f"Bifurkacje: {analysis['num_bifurcations']}")
print(f"Typy emergence: {analysis['emergence_distribution']}")
```

---

### Przykład 5: Analiza Sekwencji Tekstów

```python
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor

processor = get_gtmo_processor()

# Zbiór tekstów do analizy
texts = [
    "Einstein udowodnił, że czas jest względny.",
    "Kot śpi na macie.",
    "Być albo nie być - oto jest pytanie.",
    "Nie wiem czy może być albo jakby coś tam.",
]

print("=== ANALIZA WIELU TEKSTÓW ===\n")

for i, text in enumerate(texts, 1):
    coords, _, _ = processor.calculate_coordinates(text)

    print(f"{i}. \"{text}\"")
    print(f"   D={coords.determination:.3f}  S={coords.stability:.3f}  E={coords.entropy:.3f}")
    print()
```

---

## 🧪 Testy

### Uruchomienie Testów Morfologicznych

```bash
# Test prosty (19 testów)
python tests/test_morphology_simple.py

# Test rozszerzony
python tests/test_morphology_extended.py
```

### Test Importu

```bash
# Test czy gtmo_processor_import działa
cd Polish_Language_Processing
python gtmo_processor_import.py
```

Powinno wyświetlić:
```
🧪 Testowanie importu GTMOProcessor...
✅ GTMOProcessor zaimportowany pomyślnie!
   Typ: <class 'gtmo_morphology.GTMOProcessor'>
✅ Analiza działa!
   Współrzędne: D=0.XXX, S=0.XXX, E=0.XXX
```

### Test Dynamiki

```bash
cd Polish_Language_Processing
python -c "
import gtmo_dynamics as dyn
h = dyn.SemanticHamiltonian()
c = dyn.GTMOCoordinates(0.5, 0.5, 0.5)
print(f'Energia: {h.potential_energy(c):.4f}')
print('✅ Dynamika działa!')
"
```

---

## 🔧 Troubleshooting

### Problem 1: `ModuleNotFoundError: No module named 'Polish_Language_Processing'`

**Rozwiązanie:**
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Teraz import zadziała
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor
```

lub uruchom z katalogu głównego projektu:
```bash
cd /home/user/gtmo4nlp
python your_script.py
```

---

### Problem 2: `SyntaxError: invalid syntax` przy imporcie pliku ze spacjami

**Przyczyna:** Próbujesz zaimportować plik ze spacjami bezpośrednio.

**Rozwiązanie:** Użyj `gtmo_processor_import.py`:
```python
# ❌ NIE rób tego:
from Polish_Language_Processing.GTMØ Polish Morphological Analysis Module import GTMOProcessor

# ✅ Zrób to:
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor
processor = get_gtmo_processor()
```

---

### Problem 3: Brak modeli spaCy (HTTP 403)

**Objaw:**
```
ERROR: HTTP error 403 while getting pl_core_news_sm
```

**Rozwiązanie:**
System działa w trybie fallback bez spaCy. To **nie jest błąd krytyczny**.

Jeśli chcesz spaCy:
```bash
# Spróbuj później lub użyj lokalnej instalacji
pip install https://github.com/explosion/spacy-models/releases/download/pl_core_news_sm-3.8.0/pl_core_news_sm-3.8.0-py3-none-any.whl
```

---

### Problem 4: `matplotlib not available - visualization disabled`

**Objaw:**
```
⚠️  matplotlib not available - visualization disabled
```

**Rozwiązanie:**
```bash
pip install matplotlib
```

To dotyczy tylko wizualizacji. Wszystkie obliczenia działają bez matplotlib.

---

### Problem 5: Brak plików JSON w `data/`

**Objaw:**
```
FileNotFoundError: No such file or directory: 'data/polish_irregular_verbs.json'
```

**Rozwiązanie:**
```bash
# Sprawdź czy pliki istnieją
ls -la data/

# Jeśli nie ma, sprawdź .gitignore
cat .gitignore | grep json
```

Upewnij się że `.gitignore` zawiera:
```
!data/*.json
```

---

## 📚 Struktura Danych

### GTMOCoordinates

```python
class GTMOCoordinates:
    determination: float  # 0.0 - 1.0
    stability: float      # 0.0 - 1.0
    entropy: float        # 0.0 - 1.0
```

### SemanticTrajectory

```python
class SemanticTrajectory:
    points: List[GTMOCoordinates]  # Punkty trajektorii
    timestamps: List[float]         # Znaczniki czasu
    energies: List[float]           # Energie w każdym punkcie
    contexts: List[str]             # Konteksty tekstowe
```

---

## 🎯 Szybki Start - Krok po Kroku

### 1. Sprawdź instalację

```bash
cd /home/user/gtmo4nlp
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

### 2. Testuj import

```bash
python Polish_Language_Processing/gtmo_processor_import.py
```

### 3. Uruchom demo

```bash
python demo_gtmo_dynamics.py
```

### 4. Wypróbuj własny kod

```python
# test_gtmo.py
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor

processor = get_gtmo_processor()
coords, _, _ = processor.calculate_coordinates("Testujemy GTMØ!")

print(f"D={coords.determination:.3f}, S={coords.stability:.3f}, E={coords.entropy:.3f}")
```

```bash
python test_gtmo.py
```

---

## 📞 Wsparcie

**Repozytorium:** GSkuza/gtmo4nlp
**Branch:** `claude/discuss-project-011CUS9NNhXkjTA1MXSj4udu`

**Pliki kluczowe:**
- `Polish_Language_Processing/gtmo_processor_import.py` - helper do importu
- `demo_gtmo_dynamics.py` - kompleksowa demonstracja
- `demo_extended_features.py` - demo morfologii

**Testy:**
- `tests/test_morphology_simple.py` - podstawowe testy (19)
- `tests/test_morphology_extended.py` - testy rozszerzone

---

## 🎓 Dodatkowe Zasoby

### Dokumentacja Modułów

```python
# Zobacz dokumentację klasy
from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor
processor = get_gtmo_processor()
help(processor.calculate_coordinates)
```

### Przykładowe Pliki

- `demo_extended_features.py` - morfologia rozszerzona
- `demo_gtmo_dynamics.py` - pełna dynamika GTMØ
- `julia_sets_visualization.py` - wizualizacje zbiorów Julii (główny katalog)

### Pliki Danych

- `data/polish_irregular_verbs.json` - nieregularne czasowniki (7)
- `data/polish_derivational_affixes.json` - prefiksy i sufiksy (23)
- `data/polish_collocations.json` - kolokacje (26)
- `data/polish_semantic_similarities.json` - klastry semantyczne (7)

---

**Ostatnia aktualizacja:** 2025-10-25
**Wersja:** 1.0
**Autor:** GTMØ Framework Team

✨ Powodzenia w eksperymentach z GTMØ! ✨
