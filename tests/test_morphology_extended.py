#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testy jednostkowe dla rozszerzonych funkcji morfologicznych GTMØ
=================================================================

Testuje nowe funkcje:
- _detect_gender (rodzaj)
- _detect_number (liczba)
- _detect_verb_aspect (aspekt)
- _detect_verb_tense (czas)
- _detect_adj_degree (stopień)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Polish_Language_Processing'))

# Import modułu do testowania (tylko klasa MorfeuszAnalyzer)
import warnings
warnings.filterwarnings('ignore')

# Mock installation_status for testing
import builtins
builtins.installation_status = {
    'morfeusz': False,
    'spacy': False,
    'stanza': False
}

# Teraz importuj moduł
import importlib.util
spec = importlib.util.spec_from_file_location(
    "morphology_module",
    os.path.join(os.path.dirname(__file__), '..', 'Polish_Language_Processing',
                 'GTMØ Polish Morphological Analysis Module.py')
)
morphology_module = importlib.util.module_from_spec(spec)

# Execute the module
import logging
logging.getLogger().setLevel(logging.CRITICAL)  # Suppress logs during tests

spec.loader.exec_module(morphology_module)

# Get the MorfeuszAnalyzer class
MorfeuszAnalyzer = morphology_module.MorfeuszAnalyzer

# Testy
def test_detect_gender():
    """Test wykrywania rodzaju gramatycznego."""
    analyzer = MorfeuszAnalyzer()

    # Męskoosobowy (m1)
    assert analyzer._detect_gender("człowiek", "subst:sg:nom:m1") == "m1"
    assert analyzer._detect_gender("nauczyciel", "subst:sg:nom:m1") == "m1"

    # Żeński (f)
    assert analyzer._detect_gender("kobieta", "subst:sg:nom:f") == "f"
    assert analyzer._detect_gender("matka", "subst:sg:nom:f") == "f"
    assert analyzer._detect_gender("miłość", "subst:sg:nom:f") == "f"

    # Nijaki (n)
    assert analyzer._detect_gender("dziecko", "subst:sg:nom:n") == "n"
    assert analyzer._detect_gender("okno", "subst:sg:nom:n") == "n"

    # Męski żywotny (m2)
    assert analyzer._detect_gender("pies", "subst:sg:nom:m2") == "m2"
    assert analyzer._detect_gender("kot", "subst:sg:nom:m2") == "m2"

    print("✅ test_detect_gender: PASSED")


def test_detect_number():
    """Test wykrywania liczby gramatycznej."""
    analyzer = MorfeuszAnalyzer()

    # Liczba pojedyncza
    assert analyzer._detect_number("dom") == "sg"
    assert analyzer._detect_number("kobieta") == "sg"
    assert analyzer._detect_number("dziecko") == "sg"

    # Liczba mnoga
    assert analyzer._detect_number("domy") == "pl"
    assert analyzer._detect_number("domów") == "pl"
    assert analyzer._detect_number("domami") == "pl"
    assert analyzer._detect_number("domach") == "pl"
    assert analyzer._detect_number("ludzie") == "pl"
    assert analyzer._detect_number("dzieci") == "pl"

    print("✅ test_detect_number: PASSED")


def test_detect_verb_aspect():
    """Test wykrywania aspektu czasownika."""
    analyzer = MorfeuszAnalyzer()

    # Aspekt dokonany (perfective)
    assert analyzer._detect_verb_aspect("zrobić") == "perf"
    assert analyzer._detect_verb_aspect("powiedzieć") == "perf"
    assert analyzer._detect_verb_aspect("kupić") == "perf"

    # Aspekt niedokonany (imperfective)
    assert analyzer._detect_verb_aspect("robić") == "imperf"
    assert analyzer._detect_verb_aspect("mówić") == "imperf"
    assert analyzer._detect_verb_aspect("kupować") == "imperf"

    print("✅ test_detect_verb_aspect: PASSED")


def test_detect_verb_tense():
    """Test wykrywania czasu czasownika."""
    analyzer = MorfeuszAnalyzer()

    # Czas przeszły
    assert analyzer._detect_verb_tense("robił") == "past"
    assert analyzer._detect_verb_tense("robiłem") == "past"
    assert analyzer._detect_verb_tense("robili") == "past"

    # Czas teraźniejszy
    assert analyzer._detect_verb_tense("robię") == "pres"
    assert analyzer._detect_verb_tense("robi") == "pres"
    assert analyzer._detect_verb_tense("robią") == "pres"

    # Czas przyszły
    assert analyzer._detect_verb_tense("będę") == "fut"
    assert analyzer._detect_verb_tense("będzie") == "fut"

    print("✅ test_detect_verb_tense: PASSED")


def test_detect_adj_degree():
    """Test wykrywania stopnia przymiotnika."""
    analyzer = MorfeuszAnalyzer()

    # Stopień równy
    assert analyzer._detect_adj_degree("dobry") == "pos"
    assert analyzer._detect_adj_degree("piękny") == "pos"

    # Stopień wyższy
    assert analyzer._detect_adj_degree("lepszy") == "com"
    assert analyzer._detect_adj_degree("piękniejszy") == "com"

    # Stopień najwyższy
    assert analyzer._detect_adj_degree("najlepszy") == "sup"
    assert analyzer._detect_adj_degree("najpiękniejszy") == "sup"

    print("✅ test_detect_adj_degree: PASSED")


def test_integration():
    """Test integracyjny - analiza kompletnego zdania."""
    analyzer = MorfeuszAnalyzer()

    text = "Nauczyciel był najlepszy"
    results = analyzer.analyze(text)

    # Sprawdź czy analiza zwraca wyniki
    assert len(results) > 0
    assert results[0]['source'] == 'fallback'  # Bo Morfeusz nie jest zainstalowany w testach

    # Sprawdź strukturę wyników
    for result in results:
        assert 'form' in result
        assert 'lemma' in result
        assert 'tag' in result

    print("✅ test_integration: PASSED")
    print(f"   Przeanalizowano {len(results)} tokenów")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTY JEDNOSTKOWE - ROZSZERZONA ANALIZA MORFOLOGICZNA")
    print("="*60 + "\n")

    try:
        test_detect_gender()
        test_detect_number()
        test_detect_verb_aspect()
        test_detect_verb_tense()
        test_detect_adj_degree()
        test_integration()

        print("\n" + "="*60)
        print("✅ WSZYSTKIE TESTY ZAKOŃCZONE SUKCESEM!")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ BŁĄD KRYTYCZNY: {e}")
        import traceback
        traceback.print_exc()
