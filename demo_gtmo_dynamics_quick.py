#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Dynamics - Szybkie Demo (bez długich instalacji)
Pokazuje tylko dynamikę bez pełnej integracji z GTMOProcessor
"""

import sys
import os
sys.path.insert(0, 'Polish_Language_Processing')

import gtmo_dynamics as dyn
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def print_header(title: str):
    """Wyświetl nagłówek sekcji."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_coords(label: str, coords):
    """Wyświetl współrzędne GTMØ."""
    print(f"{label}")
    print(f"  D={coords.determination:.4f}  S={coords.stability:.4f}  E={coords.entropy:.4f}")


print("\n" + "="*70)
print("  🌌 GTMØ DYNAMICS - SZYBKIE DEMO")
print("  Demonstracja Dynamiki Semantycznej bez Długich Instalacji")
print("="*70)

# =============================================================================
# 1️⃣  DYNAMIKA HAMILTONOWSKA
# =============================================================================
print_header("1️⃣  DYNAMIKA HAMILTONOWSKA - Ewolucja Semantyczna")

hamiltonian = dyn.SemanticHamiltonian(mass=1.0, dt=0.02)

# Punkt startowy - stan wysokiej entropii
print("📍 PUNKT STARTOWY - Stan Nieokreślony")
initial = dyn.GTMOCoordinates(
    determination=0.3,
    stability=0.2,
    entropy=0.8
)
print_coords("  ", initial)
print(f"  Energia potencjalna: {hamiltonian.potential_energy(initial):.4f}")

# Ewolucja systemu
print("\n🌊 EWOLUCJA SYSTEMU...")
print("  Parametry: masa=1.0, dt=0.02, kroki=150")

trajectory = hamiltonian.evolve(initial, steps=150)

print(f"\n✅ Ewolucja zakończona!")
print(f"  • Wygenerowano {len(trajectory.points)} punktów")
print(f"  • Długość trajektorii: {trajectory.length():.4f}")
print(f"  • Średnia energia: {trajectory.mean_energy():.4f}")

print("\n📍 PUNKT KOŃCOWY")
final = trajectory.points[-1]
print_coords("  ", final)
print(f"  Energia potencjalna: {hamiltonian.potential_energy(final):.4f}")

# Zmiana współrzędnych
print("\n📊 ZMIANA WSPÓŁRZĘDNYCH:")
print(f"  ΔD = {final.determination - initial.determination:+.4f}")
print(f"  ΔS = {final.stability - initial.stability:+.4f}")
print(f"  ΔE = {final.entropy - initial.entropy:+.4f}")

# Znajdź równowagę
print("\n⚖️  SZUKANIE PUNKTU RÓWNOWAGI...")
equilibrium = hamiltonian.find_equilibrium(initial, max_iterations=500)
print_coords("  Równowaga znaleziona:", equilibrium)
print(f"  Energia w równowadze: {hamiltonian.potential_energy(equilibrium):.4f}")

# =============================================================================
# 2️⃣  ZBIORY JULII
# =============================================================================
print_header("2️⃣  ZBIORY JULII - Analiza Emergence Semantycznej")

julia = dyn.JuliaEmergence(max_iterations=100, escape_radius=2.0)

# Testowe punkty w przestrzeni GTMØ
test_points = [
    ("Wiedza Stabilna (D↑ S↑ E↓)", dyn.GTMOCoordinates(0.85, 0.80, 0.15)),
    ("Wiedza Balansowa (D~ S~ E~)", dyn.GTMOCoordinates(0.50, 0.50, 0.50)),
    ("Stan Nieokreślony (D↓ S↓ E↑)", dyn.GTMOCoordinates(0.20, 0.25, 0.85)),
    ("Emergence (D~ S↓ E~)", dyn.GTMOCoordinates(0.60, 0.35, 0.55)),
    ("Chaos (D↓ S↓ E↑)", dyn.GTMOCoordinates(0.15, 0.10, 0.95)),
]

print("🔬 ANALIZA WZORCÓW EMERGENCE:\n")

for label, coords in test_points:
    pattern = julia.emergence_pattern(coords)

    print(f"📌 {label}")
    print_coords("  ", coords)
    print(f"  ┌─ Iteracje do ucieczki: {pattern['iterations']}")
    print(f"  ├─ Stabilny w zbiorze Julii: {'TAK' if pattern['is_stable'] else 'NIE'}")
    print(f"  ├─ Typ emergence: {pattern['emergence_type']}")
    print(f"  ├─ Magntiuda końcowa: {pattern['magnitude']:.4f}")
    print(f"  └─ Prędkość ucieczki: {pattern['escape_velocity']:.4f}")
    print()

# Oblicz pole emergence dla E=0.5
print("🗺️  OBLICZANIE POLA EMERGENCE (E=0.5, rozdzielczość=40)...")
field = julia.compute_emergence_field(entropy_level=0.5, resolution=40)
print(f"  ✅ Pole wygenerowane: {field.shape}")
print(f"  • Minimum iteracji: {field.min():.0f}")
print(f"  • Maximum iteracji: {field.max():.0f}")
print(f"  • Średnia iteracji: {field.mean():.1f}")

# =============================================================================
# 3️⃣  DYNAMIKA KONTEKSTOWA (SYMULOWANA)
# =============================================================================
print_header("3️⃣  DYNAMIKA KONTEKSTOWA - Ewolucja Znaczenia (Symulowana)")

contextual = dyn.ContextualDynamicsProcessor()

# Symulowane współrzędne dla tekstów (zamiast prawdziwej analizy)
def mock_calculator(text: str) -> dyn.GTMOCoordinates:
    """Mock calculator - symuluje analizę GTMØ bez GTMOProcessor."""
    # Proste heurystyki oparte na słowach kluczowych
    text_lower = text.lower()

    # Bazowa wartość
    D, S, E = 0.5, 0.5, 0.5

    # Słowa związane z wiedzą zwiększają D i S
    if any(word in text_lower for word in ['nauka', 'wiedza', 'pewność', 'stabilność']):
        D += 0.2
        S += 0.2
        E -= 0.2

    # Słowa związane z niepewnością
    if any(word in text_lower for word in ['wątpliwość', 'niepewność', 'chaos']):
        D -= 0.2
        S -= 0.2
        E += 0.2

    # Słowa związane z emergencją
    if any(word in text_lower for word in ['emergence', 'odkrycie', 'innowacja']):
        E += 0.1

    # Klipuj do [0,1]
    D = np.clip(D, 0, 1)
    S = np.clip(S, 0, 1)
    E = np.clip(E, 0, 1)

    return dyn.GTMOCoordinates(D, S, E)

# Sekwencja kontekstów
contexts = [
    "Nauka i wiedza są fundamentem cywilizacji.",
    "Pewność i stabilność dają poczucie bezpieczeństwa.",
    "Wątpliwości prowadzą do głębszej refleksji.",
    "Niepewność otwiera drzwi do nowych odkryć.",
    "Emergence nowej wiedzy wymaga odwagi.",
    "Synteza przeciwieństw rodzi innowację."
]

print("📜 SEKWENCJA KONTEKSTÓW:\n")
for i, ctx in enumerate(contexts, 1):
    print(f"  {i}. {ctx}")

# Punkt startowy
print("\n🚀 PUNKT STARTOWY")
initial_ctx = dyn.GTMOCoordinates(0.5, 0.5, 0.5)
print_coords(f"  Neutralny stan bazowy", initial_ctx)

# Przetwarzanie sekwencji
print("\n⚙️  PRZETWARZANIE SEKWENCJI...")
trajectory_ctx = contextual.process_context_sequence(
    contexts, initial_ctx, mock_calculator
)

print(f"✅ Sekwencja przetworzona!")
print(f"  • Punktów w trajektorii: {len(trajectory_ctx.points)}")
print(f"  • Kontekstów: {len(trajectory_ctx.contexts)}")

# Analiza trajektorii
print("\n📊 ANALIZA TRAJEKTORII:")
analysis = contextual.analyze_trajectory(trajectory_ctx)

print(f"  • Całkowita odległość: {analysis['total_distance']:.4f}")
print(f"  • Średnia energia: {analysis['mean_energy']:.4f}")
print(f"  • Wariancja energii: {analysis['energy_variance']:.4f}")
print(f"  • Regionów stabilności: {analysis['num_stability_regions']}")
print(f"  • Punktów bifurkacji: {analysis['num_bifurcations']}")

if analysis['bifurcation_points']:
    print(f"  • Bifurkacje przy krokach: {analysis['bifurcation_points'][:5]}")

print(f"\n  🎭 Rozkład typów emergence:")
for etype, count in analysis['emergence_distribution'].items():
    print(f"     - {etype}: {count}")

print("\n📍 STAN KOŃCOWY")
print_coords("  ", analysis['final_coords'])

# Porównanie start vs koniec
print("\n🔄 TRANSFORMACJA SEMANTYCZNA:")
print(f"  ΔD = {analysis['final_coords'].determination - initial_ctx.determination:+.4f}")
print(f"  ΔS = {analysis['final_coords'].stability - initial_ctx.stability:+.4f}")
print(f"  ΔE = {analysis['final_coords'].entropy - initial_ctx.entropy:+.4f}")

# =============================================================================
# 4️⃣  WIELOKROTNE TRAJEKTORIE
# =============================================================================
print_header("4️⃣  PORÓWNANIE WIELU TRAJEKTORII")

print("🔬 Generowanie 5 trajektorii z różnych punktów startowych...\n")

trajectories = []
start_points = [
    ("Niska Określoność", 0.2, 0.2, 0.8),
    ("Balans", 0.5, 0.5, 0.5),
    ("Wysoka Wiedza", 0.8, 0.8, 0.2),
    ("Chaos", 0.1, 0.1, 0.9),
    ("Emergence", 0.6, 0.4, 0.6),
]

for label, d, s, e in start_points:
    start = dyn.GTMOCoordinates(d, s, e)
    traj = hamiltonian.evolve(start, steps=80)
    trajectories.append((label, traj))

    print(f"📌 {label}")
    print(f"   Start: D={d:.2f} S={s:.2f} E={e:.2f}")
    print(f"   Koniec: D={traj.points[-1].determination:.2f} " +
          f"S={traj.points[-1].stability:.2f} E={traj.points[-1].entropy:.2f}")
    print(f"   Długość: {traj.length():.4f}  Średnia energia: {traj.mean_energy():.4f}")
    print()

# =============================================================================
# 5️⃣  STATYSTYKI FINALNE
# =============================================================================
print_header("5️⃣  STATYSTYKI KOŃCOWE")

print("📊 PODSUMOWANIE WSZYSTKICH ANALIZ:\n")

print("  🎯 Dynamika Hamiltonowska:")
print(f"     • Trajektorii wygenerowano: {len(trajectories) + 1}")
print(f"     • Równowaga znaleziona w: {equilibrium}")
print(f"     • Energia w równowadze: {hamiltonian.potential_energy(equilibrium):.6f}")

print("\n  🔮 Zbiory Julii:")
print(f"     • Punktów przeanalizowano: {len(test_points)}")
print(f"     • Wszystkie stabilne: {'TAK' if all(julia.emergence_pattern(c)['is_stable'] for _, c in test_points) else 'NIE'}")
print(f"     • Pole emergence: {field.shape[0]}×{field.shape[1]} = {field.size} punktów")

print("\n  🌊 Dynamika Kontekstowa:")
print(f"     • Kontekstów przetworzono: {len(contexts)}")
print(f"     • Całkowita transformacja: {analysis['total_distance']:.4f}")
print(f"     • Bifurkacje wykryte: {analysis['num_bifurcations']}")

print("\n  📈 Porównanie Trajektorii:")
lengths = [traj.length() for _, traj in trajectories]
energies = [traj.mean_energy() for _, traj in trajectories]
print(f"     • Najdłuższa trajektoria: {max(lengths):.4f}")
print(f"     • Najkrótsza trajektoria: {min(lengths):.4f}")
print(f"     • Najwyższa energia: {max(energies):.4f}")
print(f"     • Najniższa energia: {min(energies):.4f}")

# =============================================================================
# ZAKOŃCZENIE
# =============================================================================
print_header("✅ DEMONSTRACJA ZAKOŃCZONA SUKCESEM")

print("📦 Przetestowane moduły:")
print("  ✓ SemanticHamiltonian - dynamika hamiltonowska")
print("  ✓ JuliaEmergence - analiza emergence")
print("  ✓ ContextualDynamicsProcessor - dynamika kontekstowa")
print("  ✓ DynamicsVisualizer - dostępny (wizualizacje wymagają matplotlib)")

print("\n💡 Przykład użycia w kodzie:")
print("""
from Polish_Language_Processing.gtmo_processor_import import get_hamiltonian
from Polish_Language_Processing.gtmo_dynamics import GTMOCoordinates

hamiltonian = get_hamiltonian()
coords = GTMOCoordinates(0.5, 0.5, 0.5)
trajectory = hamiltonian.evolve(coords, steps=100)

print(f"Energia: {trajectory.mean_energy():.4f}")
print(f"Długość: {trajectory.length():.4f}")
""")

print("\n🎯 Gotowe do użycia w projektach badawczych GTMØ!")
print("="*70 + "\n")
