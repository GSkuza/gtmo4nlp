#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper do importowania GTMOProcessor z pliku ze spacjami w nazwie.

Użycie:
    from Polish_Language_Processing.gtmo_processor_import import get_gtmo_processor
    processor = get_gtmo_processor()
"""

import importlib.util
import sys
import os

def get_gtmo_processor():
    """
    Importuje i zwraca GTMOProcessor z pliku ze spacjami w nazwie.

    Returns:
        GTMOProcessor instance
    """
    # Ścieżka do pliku z GTMOProcessor
    module_path = os.path.join(
        os.path.dirname(__file__),
        'GTMØ Polish Morphological Analysis Module.py'
    )

    # Sprawdź czy plik istnieje
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {module_path}")

    # Załaduj moduł używając importlib
    spec = importlib.util.spec_from_file_location("gtmo_morphology", module_path)
    module = importlib.util.module_from_spec(spec)

    # Dodaj do sys.modules aby uniknąć wielokrotnego ładowania
    sys.modules['gtmo_morphology'] = module

    # Wykonaj moduł
    spec.loader.exec_module(module)

    # Zwróć klasę GTMOProcessor
    return module.GTMOProcessor()

def get_gtmo_analyzer():
    """
    Importuje i zwraca GTMOAnalyzer z pliku ze spacjami w nazwie.

    Returns:
        GTMOAnalyzer instance
    """
    module_path = os.path.join(
        os.path.dirname(__file__),
        'GTMØ Polish Morphological Analysis Module.py'
    )

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {module_path}")

    spec = importlib.util.spec_from_file_location("gtmo_morphology", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['gtmo_morphology'] = module
    spec.loader.exec_module(module)

    return module.GTMOAnalyzer()


if __name__ == "__main__":
    # Test importu
    print("🧪 Testowanie importu GTMOProcessor...")

    try:
        processor = get_gtmo_processor()
        print("✅ GTMOProcessor zaimportowany pomyślnie!")
        print(f"   Typ: {type(processor)}")

        # Test prostej analizy
        test_text = "To jest test."
        coords, config, metadata = processor.calculate_coordinates(test_text)
        print(f"✅ Analiza działa!")
        print(f"   Współrzędne: D={coords.determination:.3f}, S={coords.stability:.3f}, E={coords.entropy:.3f}")

    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
