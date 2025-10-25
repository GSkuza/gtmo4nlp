#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo GTMØ Dynamics - Zaawansowana Analiza Dynamiki Semantycznej
================================================================

Demonstracja możliwości modułu gtmo_dynamics z integracją GTMOProcessor.

Pokazuje:
1. Dynamikę hamiltonowską dla ewolucji semantycznej
2. Analizę emergence przez zbiory Julii
3. Dynamikę kontekstową dla sekwencji tekstów
4. Wizualizacje trajektorii w przestrzeni GTMØ
5. Pełną integrację z morfologią polską

Autor: GTMØ Framework
Data: 2025-10-25
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Polish_Language_Processing'))

# Import modułów GTMØ
import warnings
warnings.filterwarnings('ignore')

# Import z pomocniczego modułu
from gtmo_processor_import import get_gtmo_processor
import gtmo_dynamics as dyn

import numpy as np


def print_header(title: str):
    """Wyświetl nagłówek sekcji."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_coords(label: str, coords):
    """Wyświetl współrzędne GTMØ."""
    print(f"{label}")
    print(f"  D={coords.determination:.4f}  S={coords.stability:.4f}  E={coords.entropy:.4f}")


def demo_1_hamiltonian_evolution():
    """Demo 1: Ewolucja hamiltonowska w przestrzeni semantycznej."""
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

    return trajectory


def demo_2_julia_emergence():
    """Demo 2: Analiza emergence przez zbiory Julii."""
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

    return julia


def demo_3_contextual_dynamics():
    """Demo 3: Dynamika kontekstowa - analiza sekwencji tekstów."""
    print_header("3️⃣  DYNAMIKA KONTEKSTOWA - Ewolucja Znaczenia")

    # Inicjalizacja procesorów
    processor = get_gtmo_processor()
    contextual = dyn.ContextualDynamicsProcessor()

    # Sekwencja kontekstów (prawdziwy polski tekst!)
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

    # Funkcja do obliczania współrzędnych z tekstu
    def calculate_coords_from_text(text: str) -> dyn.GTMOCoordinates:
        """Oblicz współrzędne GTMØ z polskiego tekstu."""
        coords, _, _ = processor.calculate_coordinates(text)
        return dyn.GTMOCoordinates(
            determination=coords.determination,
            stability=coords.stability,
            entropy=coords.entropy
        )

    # Punkt startowy
    print("\n🚀 PUNKT STARTOWY")
    initial_text = "Początek podróży."
    initial_coords = calculate_coords_from_text(initial_text)
    print_coords(f"  '{initial_text}'", initial_coords)

    # Przetwarzanie sekwencji
    print("\n⚙️  PRZETWARZANIE SEKWENCJI...")
    trajectory = contextual.process_context_sequence(
        contexts, initial_coords, calculate_coords_from_text
    )

    print(f"✅ Sekwencja przetworzona!")
    print(f"  • Punktów w trajektorii: {len(trajectory.points)}")
    print(f"  • Kontekstów: {len(trajectory.contexts)}")

    # Analiza trajektorii
    print("\n📊 ANALIZA TRAJEKTORII:")
    analysis = contextual.analyze_trajectory(trajectory)

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
    print(f"  ΔD = {analysis['final_coords'].determination - initial_coords.determination:+.4f}")
    print(f"  ΔS = {analysis['final_coords'].stability - initial_coords.stability:+.4f}")
    print(f"  ΔE = {analysis['final_coords'].entropy - initial_coords.entropy:+.4f}")

    return trajectory, analysis


def demo_4_real_polish_text():
    """Demo 4: Analiza prawdziwego polskiego tekstu."""
    print_header("4️⃣  ANALIZA PRAWDZIWYCH TEKSTÓW POLSKICH")

    processor = get_gtmo_processor()
    julia = dyn.JuliaEmergence()

    # Zbiór różnorodnych tekstów polskich
    texts = {
        "Naukowy": "Einstein udowodnił, że czas jest względny i zależy od prędkości obserwatora.",
        "Filozoficzny": "Być albo nie być - oto jest pytanie, które nurtuje ludzkość od wieków.",
        "Poetycki": "Srebrne łzy księżyca spływają po nocnym niebie cicho i delikatnie.",
        "Potoczny": "Wczoraj poszedłem do sklepu i kupiłem chleb, mleko oraz ser.",
        "Chaotyczny": "Nie wiem czy może być albo jakby coś tam różne rzeczy pewnie."
    }

    print("🔬 ANALIZA RÓŻNYCH TYPÓW TEKSTÓW:\n")

    results = []

    for category, text in texts.items():
        print(f"📝 {category.upper()}")
        print(f"   Tekst: \"{text}\"")

        # Oblicz współrzędne GTMØ
        coords_tuple, config, metadata = processor.calculate_coordinates(text)
        coords = dyn.GTMOCoordinates(
            determination=coords_tuple.determination,
            stability=coords_tuple.stability,
            entropy=coords_tuple.entropy
        )

        print_coords("   GTMØ:", coords)

        # Analiza emergence
        pattern = julia.emergence_pattern(coords)
        print(f"   Emergence: {pattern['emergence_type']}")
        print(f"   Stabilny: {'TAK' if pattern['is_stable'] else 'NIE'}")
        print(f"   Iteracje: {pattern['iterations']}")

        # Energia potencjalna
        hamiltonian = dyn.SemanticHamiltonian()
        energy = hamiltonian.potential_energy(coords)
        print(f"   Energia: {energy:.4f}")

        print()

        results.append({
            'category': category,
            'text': text,
            'coords': coords,
            'pattern': pattern,
            'energy': energy
        })

    # Podsumowanie
    print("📊 PODSUMOWANIE ANALIZY:\n")

    # Znajdź tekst najbardziej/najmniej stabilny
    stable_texts = sorted(results, key=lambda x: x['coords'].stability, reverse=True)
    print(f"  🥇 Najbardziej stabilny: {stable_texts[0]['category']}")
    print(f"     S = {stable_texts[0]['coords'].stability:.4f}")

    print(f"\n  ⚠️  Najmniej stabilny: {stable_texts[-1]['category']}")
    print(f"     S = {stable_texts[-1]['coords'].stability:.4f}")

    # Znajdź tekst o najwyższej/najniższej energii
    energy_sorted = sorted(results, key=lambda x: x['energy'], reverse=True)
    print(f"\n  ⚡ Najwyższa energia: {energy_sorted[0]['category']}")
    print(f"     E = {energy_sorted[0]['energy']:.4f}")

    print(f"\n  💤 Najniższa energia: {energy_sorted[-1]['category']}")
    print(f"     E = {energy_sorted[-1]['energy']:.4f}")

    return results


def demo_5_visualization():
    """Demo 5: Wizualizacja (jeśli matplotlib dostępny)."""
    print_header("5️⃣  WIZUALIZACJA DYNAMIKI")

    if not dyn.HAS_PLOTTING:
        print("⚠️  matplotlib nie jest dostępny - wizualizacje pominięte")
        print("   Zainstaluj matplotlib aby zobaczyć wykresy:")
        print("   pip install matplotlib")
        return None

    print("📊 Generowanie wizualizacji...\n")

    visualizer = dyn.DynamicsVisualizer()

    # 1. Trajektoria hamiltonowska
    print("1. Trajektoria 3D w przestrzeni GTMØ...")
    hamiltonian = dyn.SemanticHamiltonian()
    initial = dyn.GTMOCoordinates(0.3, 0.2, 0.7)
    traj = hamiltonian.evolve(initial, steps=100)

    try:
        visualizer.plot_trajectory_3d(
            traj,
            title="Ewolucja Hamiltonowska w GTMØ",
            save_path="gtmo_trajectory_3d.png"
        )
    except Exception as e:
        print(f"   ⚠️  Błąd: {e}")

    # 2. Ewolucja energii
    print("2. Ewolucja energii w czasie...")
    try:
        visualizer.plot_energy_evolution(
            traj,
            title="Energia Semantyczna - Ewolucja",
            save_path="gtmo_energy_evolution.png"
        )
    except Exception as e:
        print(f"   ⚠️  Błąd: {e}")

    # 3. Pole emergence
    print("3. Pole emergence Julii (E=0.5)...")
    julia = dyn.JuliaEmergence()
    try:
        visualizer.plot_emergence_field(
            julia,
            entropy_level=0.5,
            resolution=50,
            title="Pole Emergence - Zbiór Julii",
            save_path="gtmo_emergence_field.png"
        )
    except Exception as e:
        print(f"   ⚠️  Błąd: {e}")

    # 4. Portret fazowy
    print("4. Portret fazowy (D-S)...")
    try:
        # Wygeneruj kilka trajektorii
        trajectories = []
        for i in range(3):
            start = dyn.GTMOCoordinates(
                np.random.uniform(0.2, 0.8),
                np.random.uniform(0.2, 0.8),
                np.random.uniform(0.2, 0.8)
            )
            t = hamiltonian.evolve(start, steps=80)
            trajectories.append(t)

        visualizer.plot_phase_portrait(
            trajectories,
            projection='DS',
            title="Portret Fazowy GTMØ (D-S)",
            save_path="gtmo_phase_portrait.png"
        )
    except Exception as e:
        print(f"   ⚠️  Błąd: {e}")

    print("\n✅ Wizualizacje zapisane w bieżącym katalogu!")

    return visualizer


def main():
    """Główna funkcja demo."""
    print("\n" + "="*70)
    print("  🌌 GTMØ DYNAMICS - KOMPLEKSOWA DEMONSTRACJA")
    print("  Zaawansowana Analiza Dynamiki Semantycznej dla Języka Polskiego")
    print("="*70)

    try:
        # Demo 1: Hamiltonowska dynamika
        traj1 = demo_1_hamiltonian_evolution()

        # Demo 2: Zbiory Julii
        julia = demo_2_julia_emergence()

        # Demo 3: Dynamika kontekstowa
        traj2, analysis = demo_3_contextual_dynamics()

        # Demo 4: Prawdziwe polskie teksty
        results = demo_4_real_polish_text()

        # Demo 5: Wizualizacja
        viz = demo_5_visualization()

        # Podsumowanie końcowe
        print_header("✅ DEMONSTRACJA ZAKOŃCZONA SUKCESEM")

        print("📦 Moduły przygotowane:")
        print("  • SemanticHamiltonian - dynamika hamiltonowska")
        print("  • JuliaEmergence - analiza emergence")
        print("  • ContextualDynamicsProcessor - dynamika kontekstowa")
        print("  • DynamicsVisualizer - wizualizacje")
        print("  • GTMOProcessor - morfologia polska (zintegrowana)")

        print("\n💡 Przykłady użycia:")
        print("  from Polish_Language_Processing.gtmo_processor_import import get_hamiltonian")
        print("  hamiltonian = get_hamiltonian()")
        print("  trajectory = hamiltonian.evolve(initial_coords, steps=100)")

        print("\n🎯 Gotowe do użycia w projektach badawczych!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
