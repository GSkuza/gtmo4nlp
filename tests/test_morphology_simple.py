#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proste testy dla nowych funkcji morfologicznych (bez pełnej instalacji)
"""

# Symulacja klas i funkcji do testowania
class SimpleMorfeuszAnalyzer:
    """Minimalna wersja analyzera do testów."""

    def _detect_gender(self, word: str, tag: str) -> str:
        """Rozpoznaj rodzaj gramatyczny słowa (m1/m2/m3/f/n)."""
        word_lower = word.lower()

        # m1 (męskoosobowy) - osoby płci męskiej
        m1_endings = ['ek', 'ik', 'arz', 'cz', 'nik', 'ak', 'ec']
        m1_words = ['człowiek', 'mężczyzna', 'chłopiec', 'ojciec', 'syn', 'brat',
                    'dziadek', 'wuj', 'kuzyn', 'kolega', 'przyjaciel', 'nauczyciel']

        if word_lower in m1_words:
            return 'm1'
        for ending in m1_endings:
            if word_lower.endswith(ending) and len(word_lower) > len(ending) + 2:
                return 'm1'

        # f (żeński)
        f_endings = ['a', 'ka', 'ia', 'ość', 'ja', 'ni', 'yni', 'owa', 'wa']
        f_words = ['kobieta', 'matka', 'córka', 'siostra', 'babcia', 'ciocia']

        if word_lower in f_words:
            return 'f'
        for ending in f_endings:
            if word_lower.endswith(ending):
                return 'f'

        # n (nijaki)
        n_endings = ['o', 'um', 'ę', 'cie', 'nie', 'dło', 'tko']
        n_words = ['dziecko', 'pole', 'morze', 'okno', 'miasto']

        if word_lower in n_words:
            return 'n'
        for ending in n_endings:
            if word_lower.endswith(ending):
                return 'n'

        # m2 (męski żywotny) - zwierzęta
        m2_words = ['pies', 'kot', 'koń', 'ptak', 'słoń', 'lew', 'wilk']
        if word_lower in m2_words:
            return 'm2'

        # m3 (męski nieżywotny) - rzeczy, domyślny dla rzeczowników męskich
        return 'm3'

    def _detect_number(self, word: str) -> str:
        """Rozpoznaj liczbę gramatyczną (sg/pl)."""
        word_lower = word.lower()

        # Liczba mnoga - końcówki
        pl_endings = ['y', 'i', 'ów', 'om', 'ami', 'mi', 'ach']
        pl_words = ['ludzie', 'dzieci', 'oczy', 'ręce', 'nogi']

        if word_lower in pl_words:
            return 'pl'

        for ending in pl_endings:
            if word_lower.endswith(ending) and len(word_lower) > len(ending) + 2:
                if ending in ['y', 'i'] and len(word_lower) > 3:
                    return 'pl'
                elif ending not in ['y', 'i']:
                    return 'pl'

        return 'sg'

    def _detect_verb_aspect(self, lemma: str) -> str:
        """Rozpoznaj aspekt czasownika (perf/imperf)."""
        lemma_lower = lemma.lower()

        perfective_prefixes = ['z', 'wy', 'prze', 'roz', 'po', 'na', 'u', 's', 'do',
                              'od', 'przy', 'we', 'za', 'ob']

        for prefix in perfective_prefixes:
            if lemma_lower.startswith(prefix) and len(lemma_lower) > len(prefix) + 2:
                if lemma_lower.startswith(prefix + 'r') or lemma_lower.startswith(prefix + 'p'):
                    return 'perf'

        imperfective_suffixes = ['ywać', 'iwać', 'ować']
        for suffix in imperfective_suffixes:
            if lemma_lower.endswith(suffix):
                return 'imperf'

        perfective_verbs = ['zrobić', 'powiedzieć', 'wziąć', 'dać', 'kupić', 'sprzedać']
        imperfective_verbs = ['robić', 'mówić', 'brać', 'dawać', 'kupować', 'sprzedawać']

        if lemma_lower in perfective_verbs:
            return 'perf'
        elif lemma_lower in imperfective_verbs:
            return 'imperf'

        return 'imperf'

    def _detect_verb_tense(self, word: str) -> str:
        """Rozpoznaj czas czasownika (pres/past/fut)."""
        word_lower = word.lower()

        past_endings = ['łem', 'łeś', 'ł', 'ła', 'ło', 'liśmy', 'łyśmy', 'li', 'ły']
        for ending in past_endings:
            if word_lower.endswith(ending):
                return 'past'

        future_forms = ['będę', 'będziesz', 'będzie', 'będziemy', 'będziecie', 'będą']
        if word_lower in future_forms:
            return 'fut'

        present_endings = ['ę', 'esz', 'e', 'emy', 'ecie', 'ą', 'am', 'asz', 'amy', 'acie', 'ają']
        for ending in present_endings:
            if word_lower.endswith(ending):
                return 'pres'

        return 'pres'

    def _detect_adj_degree(self, word: str) -> str:
        """Rozpoznaj stopień przymiotnika (pos/com/sup)."""
        word_lower = word.lower()

        if word_lower.startswith('naj'):
            return 'sup'

        comparative_endings = ['szy', 'ejszy', 'iejszy']
        for ending in comparative_endings:
            if word_lower.endswith(ending):
                return 'com'

        return 'pos'


# Testy
def run_tests():
    analyzer = SimpleMorfeuszAnalyzer()

    print("\n" + "="*60)
    print("PROSTE TESTY JEDNOSTKOWE - MORFOLOGIA")
    print("="*60 + "\n")

    passed = 0
    failed = 0

    # Test 1: Rodzaj
    print("Test 1: Wykrywanie rodzaju...")
    tests = [
        ("człowiek", "m1"),
        ("kobieta", "f"),
        ("dziecko", "n"),
        ("pies", "m2"),
        ("dom", "m3"),
    ]
    for word, expected in tests:
        result = analyzer._detect_gender(word, "")
        if result == expected:
            print(f"  ✅ {word} → {result}")
            passed += 1
        else:
            print(f"  ❌ {word} → {result} (expected {expected})")
            failed += 1

    # Test 2: Liczba
    print("\nTest 2: Wykrywanie liczby...")
    tests = [
        ("dom", "sg"),
        ("domy", "pl"),
        ("domów", "pl"),
        ("ludzie", "pl"),
    ]
    for word, expected in tests:
        result = analyzer._detect_number(word)
        if result == expected:
            print(f"  ✅ {word} → {result}")
            passed += 1
        else:
            print(f"  ❌ {word} → {result} (expected {expected})")
            failed += 1

    # Test 3: Aspekt
    print("\nTest 3: Wykrywanie aspektu...")
    tests = [
        ("zrobić", "perf"),
        ("robić", "imperf"),
        ("kupować", "imperf"),
        ("kupić", "perf"),
    ]
    for word, expected in tests:
        result = analyzer._detect_verb_aspect(word)
        if result == expected:
            print(f"  ✅ {word} → {result}")
            passed += 1
        else:
            print(f"  ❌ {word} → {result} (expected {expected})")
            failed += 1

    # Test 4: Czas
    print("\nTest 4: Wykrywanie czasu...")
    tests = [
        ("robił", "past"),
        ("robię", "pres"),
        ("będę", "fut"),
    ]
    for word, expected in tests:
        result = analyzer._detect_verb_tense(word)
        if result == expected:
            print(f"  ✅ {word} → {result}")
            passed += 1
        else:
            print(f"  ❌ {word} → {result} (expected {expected})")
            failed += 1

    # Test 5: Stopień
    print("\nTest 5: Wykrywanie stopnia przymiotnika...")
    tests = [
        ("dobry", "pos"),
        ("lepszy", "com"),
        ("najlepszy", "sup"),
    ]
    for word, expected in tests:
        result = analyzer._detect_adj_degree(word)
        if result == expected:
            print(f"  ✅ {word} → {result}")
            passed += 1
        else:
            print(f"  ❌ {word} → {result} (expected {expected})")
            failed += 1

    # Podsumowanie
    print("\n" + "="*60)
    print(f"WYNIKI: {passed} PASSED, {failed} FAILED")
    if failed == 0:
        print("✅ WSZYSTKIE TESTY ZAKOŃCZONE SUKCESEM!")
    else:
        print(f"❌ {failed} testów nie powiodło się")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
