#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Polish NLP Implementation - NAPRAWIONA WERSJA
==================================================
Kompletna implementacja GTMØ dla języka polskiego z naprawioną instalacją
optimized for Google Colab environment.

Author: GTMØ Research Team
Version: 2.1 - FIXED Installation Issues
Date: 2024
"""

import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NAPRAWIONA INSTALACJA ZALEŻNOŚCI
# ============================================================================

def install_dependencies():
    """Naprawiona instalacja z lepszą obsługą błędów."""
    print("🔧 Konfiguracja środowiska Google Colab...")

    # Lista podstawowych pakietów
    basic_packages = [
        'plotly',
        'kaleido',  # For plotly export
        'pandas',
        'numpy'
    ]

    # Zainstaluj podstawowe pakiety
    for package in basic_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} już zainstalowany")
        except ImportError:
            print(f"📦 Instaluję {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✅ {package} zainstalowany")

    # Spróbuj zainstalować Morfeusz2 z kilkoma metodami
    morfeusz_installed = install_morfeusz2()

    # Zainstaluj spaCy i model polski
    spacy_installed = install_spacy_polish()

    # Zainstaluj Stanza (opcjonalnie)
    stanza_installed = install_stanza_polish()

    print("\n📊 Podsumowanie instalacji:")
    print(f"   📦 Podstawowe pakiety: ✅")
    print(f"   🔤 Morfeusz2: {'✅' if morfeusz_installed else '❌ (będzie używany fallback)'}")
    print(f"   🧠 spaCy: {'✅' if spacy_installed else '❌ (będzie używany fallback)'}")
    print(f"   📚 Stanza: {'✅' if stanza_installed else '❌ (będzie używany fallback)'}")

    return {
        'morfeusz': morfeusz_installed,
        'spacy': spacy_installed,
        'stanza': stanza_installed
    }

def install_morfeusz2():
    """Naprawiona instalacja Morfeusz2 z wieloma metodami."""
    print("🔤 Próbuję zainstalować Morfeusz2...")

    # Metoda 1: Próba standardowej instalacji
    try:
        import morfeusz2
        print("✅ Morfeusz2 już dostępny")
        return True
    except ImportError:
        pass

    # Metoda 2: pip install morfeusz2
    try:
        print("   📦 Metoda 1: pip install morfeusz2...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "morfeusz2"])
        import morfeusz2
        print("✅ Morfeusz2 zainstalowany przez pip")
        return True
    except:
        print("   ❌ Metoda 1 nieudana")

    # Metoda 3: Bezpośredni wheel
    wheel_urls = [
        "https://github.com/morfeusz-project/morfeusz/releases/download/2.1.9/morfeusz2-2.1.9-py3-none-any.whl",
        "https://github.com/morfeusz-project/morfeusz/releases/download/2.0.9/morfeusz2-2.0.9-py3-none-any.whl"
    ]

    for url in wheel_urls:
        try:
            print(f"   📦 Metoda 2: Próbuję {url}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", url])
            import morfeusz2
            print("✅ Morfeusz2 zainstalowany z wheel")
            return True
        except Exception as e:
            print(f"   ❌ Błąd: {str(e)[:100]}")
            continue

    # Metoda 4: Instalacja zależności systemowych
    try:
        print("   📦 Metoda 3: Instaluję zależności systemowe...")
        subprocess.check_call(["apt-get", "update", "-qq"])
        subprocess.check_call(["apt-get", "install", "-y", "-qq", "python3-dev", "build-essential"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "morfeusz2"])
        import morfeusz2
        print("✅ Morfeusz2 zainstalowany z zależnościami")
        return True
    except:
        print("   ❌ Metoda 3 nieudana")

    print("⚠️ Morfeusz2 niedostępny - system będzie używał fallback")
    return False

def install_spacy_polish():
    """Instalacja spaCy z polskim modelem."""
    print("🧠 Instaluję spaCy...")

    try:
        import spacy
        # Sprawdź czy model polski jest dostępny
        try:
            nlp = spacy.load("pl_core_news_sm")
            print("✅ spaCy z modelem polskim już dostępny")
            return True
        except:
            pass
    except ImportError:
        # Zainstaluj spaCy
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "spacy"])
            import spacy
        except:
            print("❌ Nie można zainstalować spaCy")
            return False

    # Zainstaluj model polski
    models_to_try = ["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"]

    for model in models_to_try:
        try:
            print(f"   📚 Pobieram model {model}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model, "--quiet"])

            # Test czy model działa
            import spacy
            nlp = spacy.load(model)
            print(f"✅ spaCy model {model} zainstalowany")
            return True
        except Exception as e:
            print(f"   ❌ Model {model} nieudany: {str(e)[:50]}")
            continue

    print("⚠️ spaCy zainstalowany ale bez polskiego modelu")
    return False

def install_stanza_polish():
    """Instalacja Stanza (opcjonalna)."""
    print("📚 Instaluję Stanza...")

    try:
        # Zainstaluj Stanza
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "stanza"])

        # Sprawdź czy działa
        import stanza
        print("✅ Stanza zainstalowana (model zostanie pobrany przy pierwszym użyciu)")
        return True
    except Exception as e:
        print(f"❌ Stanza installation failed: {str(e)[:50]}")
        return False

# Uruchom instalację na początku komórki
print("=" * 60)
print("GTMØ POLISH NLP - NAPRAWIONA INSTALACJA")
print("=" * 60)

# Store installation status globally for access in other parts of the script
global installation_status
installation_status = install_dependencies()


# ============================================================================
# IMPORTY I KONFIGURACJA
# ============================================================================

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly niedostępny - wizualizacje wyłączone")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PODSTAWOWE KLASY GTMØ
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
SQRT_2_INV = 1 / np.sqrt(2)  # Quantum amplitude
COGNITIVE_CENTER = np.array([0.5, 0.5, 0.5])  # Neutral knowledge state

# ============================================================================
# GTMØ WSPÓŁRZĘDNE DLA CECH MORFOLOGICZNYCH
# ============================================================================

# Aspekt czasownika (determination, stability, entropy)
ASPECT_COORDS = {
    'perf': (0.8, 0.7, 0.2),    # Dokonany = określony, zakończony
    'imperf': (0.5, 0.5, 0.5)   # Niedokonany = trwały, niezakończony
}

# Czas czasownika
TENSE_COORDS = {
    'past': (0.7, 0.8, 0.2),    # Przeszły = stabilny, określony
    'pres': (0.6, 0.5, 0.4),    # Teraźniejszy = mniej stabilny
    'fut': (0.4, 0.3, 0.6)      # Przyszły = niepewny, wysoka entropia
}

# Liczba gramatyczna
NUMBER_COORDS = {
    'sg': (0.7, 0.7, 0.3),      # Pojedyncza = bardziej określona
    'pl': (0.5, 0.5, 0.5)       # Mnoga = bardziej ogólna
}

# Rodzaj gramatyczny
GENDER_COORDS = {
    'm1': (0.8, 0.7, 0.3),      # Męskoosobowy = określony
    'm2': (0.7, 0.6, 0.4),      # Męski żywotny
    'm3': (0.6, 0.6, 0.4),      # Męski nieżywotny
    'f': (0.7, 0.7, 0.3),       # Żeński = określony
    'n': (0.6, 0.5, 0.5)        # Nijaki = mniej określony
}

# Stopień przymiotnika
DEGREE_COORDS = {
    'pos': (0.6, 0.7, 0.3),     # Równy = normalny stopień
    'com': (0.5, 0.5, 0.5),     # Wyższy = porównawczy
    'sup': (0.8, 0.6, 0.4)      # Najwyższy = najbardziej określony
}

@dataclass
class GTMOCoordinates:
    """Współrzędne GTMØ w przestrzeni fazowej 3D."""
    determination: float = 0.5
    stability: float = 0.5
    entropy: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([self.determination, self.stability, self.entropy])

    def distance_to(self, other: 'GTMOCoordinates') -> float:
        return np.linalg.norm(self.to_array() - other.to_array())

@dataclass
class Configuration:
    """Kompletna konfiguracja w Przestrzeni Konfiguracyjnej GTMØ."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0
    orientation: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    scale: float = 1.0

    def to_dict(self) -> Dict:
        return {
            'position': self.position.tolist(),
            'time': self.time,
            'orientation': self.orientation.tolist(),
            'scale': self.scale
        }

class PolishCase(Enum):
    """Polskie przypadki gramatyczne z współrzędnymi GTMØ."""
    NOMINATIVE = ("nom", "mianownik", 0.901, 0.799, 0.151)
    GENITIVE = ("gen", "dopełniacz", 0.701, 0.751, 0.301)
    DATIVE = ("dat", "celownik", 0.651, 0.549, 0.451)
    ACCUSATIVE = ("acc", "biernik", 0.851, 0.801, 0.201)
    INSTRUMENTAL = ("inst", "narzędnik", 0.499, 0.301, 0.899)
    LOCATIVE = ("loc", "miejscownik", 0.501, 0.499, 0.801)
    VOCATIVE = ("voc", "wołacz", 0.149, 0.151, 0.851)

    def __init__(self, tag: str, polish_name: str,
                 determination: float, stability: float, entropy: float):
        self.tag = tag
        self.polish_name = polish_name
        self.coords = GTMOCoordinates(determination, stability, entropy)

class PolishPOS(Enum):
    """Polish parts of speech with GTMØ coordinates."""
    SUBST = ("subst", "rzeczownik", 0.8, 0.9, 0.2)
    ADJ = ("adj", "przymiotnik", 0.6, 0.5, 0.4)
    ADV = ("adv", "przysłówek", 0.5, 0.6, 0.4)
    VERB = ("verb", "czasownik", 0.7, 0.4, 0.5)
    NUM = ("num", "liczebnik", 0.9, 0.8, 0.1)
    PRON = ("pron", "zaimek", 0.8, 0.6, 0.3)
    PREP = ("prep", "przyimek", 0.6, 0.8, 0.3)
    CONJ = ("conj", "spójnik", 0.5, 0.7, 0.4)
    PART = ("part", "partykuła", 0.3, 0.2, 0.8)
    INTERP = ("interp", "interpunkcja", 0.9, 0.9, 0.1)

    def __init__(self, tag: str, polish_name: str,
                 determination: float, stability: float, entropy: float):
        self.tag = tag
        self.polish_name = polish_name
        self.coords = GTMOCoordinates(determination, stability, entropy)

class KnowledgeAttractor(Enum):
    """Atraktory wiedzy GTMØ do klasyfikacji."""
    PARTICLE = ("Ψᴷ", "Knowledge Particle", 0.85, 0.85, 0.15)
    SHADOW = ("Ψʰ", "Knowledge Shadow", 0.15, 0.15, 0.85)
    EMERGENT = ("Ψᴺ", "Emergent", 0.5, 0.3, 0.9)
    SINGULARITY = ("Ø", "Singularity", 1.0, 1.0, 0.0)
    FLUX = ("Ψ~", "Flux", 0.5, 0.5, 0.8)
    VOID = ("Ψ◊", "Void", 0.0, 0.0, 0.5)

    def __init__(self, symbol: str, display_name: str,
                 determination: float, stability: float, entropy: float):
        self.symbol = symbol
        self.display_name = display_name
        self.coords = GTMOCoordinates(determination, stability, entropy)

# ============================================================================
# ANALIZATOR MORFOLOGICZNY Z FALLBACK
# ============================================================================

class MorfeuszAnalyzer:
    """Analizator morfologiczny Morfeusz2 z fallback - TYLKO MORFOLOGIA."""

    def __init__(self):
        # Check global installation status
        if installation_status.get('morfeusz'):
            try:
                import morfeusz2
                self.morfeusz = morfeusz2.Morfeusz()
                self.available = True
                logger.info("✅ Morfeusz2 zainicjalizowany")
            except Exception as e:
                self.morfeusz = None
                self.available = False
                logger.warning(f"❌ Morfeusz2 błąd: {e}")
        else:
            self.morfeusz = None
            self.available = False
            logger.info("ℹ️ Morfeusz2 niedostępny - używam fallback")

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Analiza morfologiczna z fallback."""
        if not self.available:
            return self._fallback_analysis(text)

        try:
            analysis = self.morfeusz.analyse(text)
            results = []

            for segment in analysis:
                start_idx, end_idx, interpretation = segment
                form, lemma, tag, labels, qualifiers = interpretation

                results.append({
                    'form': form,
                    'lemma': lemma,
                    'tag': tag,
                    'labels': labels if labels else [],
                    'qualifiers': qualifiers if qualifiers else [],
                    'start': start_idx,
                    'end': end_idx,
                    'source': 'morfeusz2'
                })

            return results
        except Exception as e:
            logger.warning(f"Morfeusz błąd: {e}, używam fallback")
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> List[Dict[str, Any]]:
        """Prosta analiza morfologiczna fallback dla polskiego."""
        words = text.split()
        results = []

        for i, word in enumerate(words):
            # Prosta heurystyka morfologiczna dla polskiego
            lemma = self._guess_lemma(word)
            tag = self._guess_morphological_tag(word)

            results.append({
                'form': word,
                'lemma': lemma,
                'tag': tag,
                'labels': [],
                'qualifiers': [],
                'start': i,
                'end': i + 1,
                'source': 'fallback'
            })

        return results

    def _guess_lemma(self, word: str) -> str:
        """Prosta heurystyka lematyzacji."""
        word_lower = word.lower()

        # Usuwanie prostych końcówek fleksyjnych
        if word_lower.endswith(('em', 'ie', 'ą', 'y', 'e')):
            return word_lower[:-1]
        elif word_lower.endswith(('ami', 'ach', 'ów')):
            return word_lower[:-2]
        else:
            return word_lower

    def _guess_morphological_tag(self, word: str) -> str:
        """Rozszerzona heurystyka tagów morfologicznych."""
        word_lower = word.lower()

        # Czasowniki
        if word_lower.endswith(('ać', 'eć', 'ić', 'ować', 'nąć')):
            return 'verb:inf'
        elif word_lower.endswith(('ę', 'esz', 'e', 'emy', 'ecie', 'ą')):
            return 'verb:fin:sg:3:pres'

        # Rzeczowniki
        elif word_lower.endswith(('ość', 'enie', 'anie', 'cie')):
            return 'subst:sg:nom:f'
        elif word_lower.endswith(('ek', 'ik', 'ar', 'arz')):
            return 'subst:sg:nom:m1'
        elif word_lower.endswith(('a', 'ka', 'ia')):
            return 'subst:sg:nom:f'
        elif word_lower.endswith(('o', 'um', 'e')):
            return 'subst:sg:nom:n'

        # Przymiotniki
        elif word_lower.endswith(('ny', 'ty', 'ły', 'wy', 'owy', 'owy')):
            return 'adj:sg:nom:m1:pos'
        elif word_lower.endswith(('na', 'ta', 'ła', 'wa', 'owa')):
            return 'adj:sg:nom:f:pos'
        elif word_lower.endswith(('ne', 'te', 'łe', 'we', 'owe')):
            return 'adj:sg:nom:n:pos'

        # Przysłówki
        elif word_lower.endswith(('sko', 'śnie', 'ąco', 'nie')):
            return 'adv:pos'

        # Spójniki i partykuły
        elif word_lower in ['i', 'a', 'ale', 'że', 'bo', 'oraz', 'lub']:
            return 'conj'
        elif word_lower in ['nie', 'tak', 'już', 'jeszcze', 'bardzo']:
            return 'part'

        # Przyimki
        elif word_lower in ['w', 'na', 'do', 'z', 'o', 'od', 'po', 'za', 'przed']:
            return 'prep:inst'

        # Domyślnie rzeczownik
        else:
            return 'subst:sg:nom:m3'

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
                # Heurystyka: jeśli ma typową końcówkę męskoosobową
                return 'm1'

        # f (żeński)
        f_endings = ['a', 'ka', 'ia', 'ość', 'ość', 'ja', 'ni', 'yni', 'owa', 'wa']
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
        pl_endings = [
            'y', 'i',           # nom.pl: domy, konie
            'ów', 'i',          # gen.pl: domów, koni
            'om',               # dat.pl: domom
            'ami', 'mi',        # inst.pl: domami
            'ach',              # loc.pl: domach
        ]

        # Wyjątki liczby mnogiej
        pl_words = ['ludzie', 'dzieci', 'oczy', 'ręce', 'nogi']
        if word_lower in pl_words:
            return 'pl'

        # Sprawdź końcówki liczby mnogiej
        for ending in pl_endings:
            if word_lower.endswith(ending) and len(word_lower) > len(ending) + 2:
                # Dodatkowa heurystyka: unikaj fałszywych trafień dla krótkich słów
                if ending in ['y', 'i'] and len(word_lower) > 3:
                    return 'pl'
                elif ending not in ['y', 'i']:
                    return 'pl'

        # Domyślnie liczba pojedyncza
        return 'sg'

    def _detect_verb_aspect(self, lemma: str) -> str:
        """Rozpoznaj aspekt czasownika (perf/imperf)."""
        lemma_lower = lemma.lower()

        # Prefiksy dokonane (perfektywne)
        perfective_prefixes = [
            'z', 'wy', 'prze', 'roz', 'po', 'na', 'u', 's', 'do',
            'od', 'przy', 'we', 'za', 'ob'
        ]

        for prefix in perfective_prefixes:
            if lemma_lower.startswith(prefix) and len(lemma_lower) > len(prefix) + 2:
                # Sprawdź czy to faktycznie prefiks a nie część rdzenia
                if lemma_lower.startswith(prefix + 'r') or lemma_lower.startswith(prefix + 'p'):
                    return 'perf'

        # Sufiksy niedokonane (imperfektywne)
        imperfective_suffixes = ['ywać', 'iwać', 'ować']
        for suffix in imperfective_suffixes:
            if lemma_lower.endswith(suffix):
                return 'imperf'

        # Typowe czasowniki dokonane i niedokonane (słownik)
        perfective_verbs = ['zrobić', 'powiedzieć', 'wziąć', 'dać', 'kupić', 'sprzedać']
        imperfective_verbs = ['robić', 'mówić', 'brać', 'dawać', 'kupować', 'sprzedawać']

        if lemma_lower in perfective_verbs:
            return 'perf'
        elif lemma_lower in imperfective_verbs:
            return 'imperf'

        # Domyślnie niedokonany
        return 'imperf'

    def _detect_verb_tense(self, word: str) -> str:
        """Rozpoznaj czas czasownika (pres/past/fut)."""
        word_lower = word.lower()

        # Czas przeszły - końcówki
        past_endings = ['łem', 'łeś', 'ł', 'ła', 'ło', 'liśmy', 'łyśmy', 'li', 'ły']
        for ending in past_endings:
            if word_lower.endswith(ending):
                return 'past'

        # Czas przyszły - formy analityczne z 'będę'
        future_forms = ['będę', 'będziesz', 'będzie', 'będziemy', 'będziecie', 'będą']
        if word_lower in future_forms:
            return 'fut'

        # Czas teraźniejszy - końcówki (domyślny)
        present_endings = ['ę', 'esz', 'e', 'emy', 'ecie', 'ą', 'am', 'asz', 'amy', 'acie', 'ają']
        for ending in present_endings:
            if word_lower.endswith(ending):
                return 'pres'

        # Domyślnie teraźniejszy
        return 'pres'

    def _detect_adj_degree(self, word: str) -> str:
        """Rozpoznaj stopień przymiotnika (pos/com/sup)."""
        word_lower = word.lower()

        # Stopień najwyższy - prefiks 'naj-'
        if word_lower.startswith('naj'):
            return 'sup'

        # Stopień wyższy - końcówki
        comparative_endings = ['szy', 'ejszy', 'iejszy']
        for ending in comparative_endings:
            if word_lower.endswith(ending):
                return 'com'

        # Stopień równy (domyślny)
        return 'pos'

class SpacyAnalyzer:
    """spaCy analyzer for Polish syntactic analysis with fallback."""

    def __init__(self):
         # Check global installation status
        if installation_status.get('spacy'):
            try:
                import spacy
                self.nlp = spacy.load("pl_core_news_lg") # Try lg first
                self.available = True
                logger.info("✅ spaCy Polish model loaded")
            except:
                 try:
                     self.nlp = spacy.load("pl_core_news_md") # Try md
                     self.available = True
                     logger.info("✅ spaCy Polish model loaded (md)")
                 except:
                     try:
                         self.nlp = spacy.load("pl_core_news_sm") # Try sm
                         self.available = True
                         logger.info("✅ spaCy Polish model loaded (sm)")
                     except Exception as e:
                         self.nlp = None
                         self.available = False
                         logger.warning(f"❌ spaCy not available: {e}")
        else:
            self.nlp = None
            self.available = False
            logger.info("ℹ️ spaCy niedostępny - używam fallback")


    def analyze(self, text: str) -> Optional[Any]:
        """Analyze text with spaCy or fallback."""
        if not self.available:
            return self._fallback_analysis(text)

        try:
            return self.nlp(text)
        except Exception as e:
            logger.warning(f"spaCy błąd: {e}, używam fallback")
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> Optional[Any]:
        """Prosta analiza fallback spaCy (syntaktyka)."""
        # Create a dummy Doc-like object for fallback
        class DummyMorph:
            """Dummy class to mimic spaCy Morph object with .get method returning a list."""
            def __init__(self, data: Dict = None):
                self._data = data if data is not None else {}

            def get(self, key, default=None):
                # Return a list or default, mimicking spaCy's behavior
                return [self._data[key]] if key in self._data else default


        class DummyToken:
            def __init__(self, text, pos, dep, lemma, morph_data: Dict = None, is_stop=False, head=None, children=None):
                self.text = text
                self.pos_ = pos
                self.dep_ = dep
                self.lemma_ = lemma
                # Initialize morph as a DummyMorph object
                self.morph = DummyMorph(morph_data)
                self.is_stop = is_stop
                self.head = head if head is not None else self
                self.children = children if children is not None else []
                self.has_vector = False
                self.vector = None

            def __str__(self):
                return self.text

            @property
            def ancestors(self):
                # Simplified: assume max depth 2 for fallback
                return [self.head] if self.head != self else []


        class DummyDoc:
            def __init__(self, text):
                self.text = text
                words = text.split()
                self.tokens = []
                root_found = False
                for i, word in enumerate(words):
                    # Simple heuristic for POS and dependency
                    pos = 'X'
                    dep = 'dep'
                    lemma = word.lower()
                    morph_data = {} # Default empty morph data

                    if word.lower() in ['jest', 'są', 'być']:
                        pos = 'VERB'
                        dep = 'ROOT'
                        root_found = True
                        # Add dummy morph data for verb
                        morph_data = {"Aspect": ["Imp"], "Mood": ["Ind"]}
                    elif word.lower() in ['i', 'a', 'ale']:
                         pos = 'CCONJ'
                    elif word.lower() in ['.', ',', '!', '?']:
                         pos = 'PUNCT'

                    # Pass dummy morph data
                    self.tokens.append(DummyToken(word, pos, dep, lemma, morph_data=morph_data))

                # Basic dependency linkage for fallback
                if root_found:
                    root_token = next((t for t in self.tokens if t.dep_ == 'ROOT'), None)
                    if root_token:
                        for token in self.tokens:
                            if token != root_token:
                                token.head = root_token # Link all others to root


            def __iter__(self):
                return iter(self.tokens)

            def __len__(self):
                return len(self.tokens)

        logger.info("Using spaCy fallback analysis.")
        return DummyDoc(text)

class StanzaAnalyzer:
    """Stanza analyzer for Polish with fallback."""

    def __init__(self):
         # Check global installation status
        if installation_status.get('stanza'):
            try:
                import stanza
                # Download model if needed (handled in install_dependencies)
                self.nlp = stanza.Pipeline('pl',
                                        processors='tokenize,pos,lemma,depparse',
                                        verbose=False,
                                        use_gpu=False)
                self.available = True
                logger.info("✅ Stanza Polish pipeline loaded")
            except Exception as e:
                self.nlp = None
                self.available = False
                logger.warning(f"❌ Stanza not available: {e}")
        else:
            self.nlp = None
            self.available = False
            logger.info("ℹ️ Stanza niedostępny - używam fallback")


    def analyze(self, text: str) -> Optional[Any]:
        """Analyze text with Stanza or fallback."""
        if not self.available:
            return self._fallback_analysis(text)

        try:
            doc = self.nlp(text)
            return doc
        except Exception as e:
            logger.warning(f"Stanza błąd: {e}, używam fallback")
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> Optional[Any]:
        """Prosta analiza fallback Stanza."""
        logger.info("Using Stanza fallback analysis.")
        # Return a simple structure, possibly mirroring the Stanza Doc structure partially
        class DummyStanzaWord:
            def __init__(self, text, lemma, upos, xpos, feats, head, deprel):
                self.text = text
                self.lemma = lemma
                self.upos = upos
                self.xpos = xpos
                self.feats = feats
                self.head = head
                self.deprel = deprel

        class DummyStanzaSentence:
            def __init__(self, text):
                self.text = text
                words = text.split()
                self.words = []
                for word in words:
                    # Simple heuristic
                    lemma = word.lower()
                    upos = 'X'
                    xpos = '_'
                    feats = '_'
                    head = 0 # Simplified head (0 for root)
                    deprel = 'root' if head == 0 else 'dep'
                    self.words.append(DummyStanzaWord(word, lemma, upos, xpos, feats, head, deprel))

        class DummyStanzaDoc:
            def __init__(self, text):
                self.text = text
                self.sentences = [DummyStanzaSentence(text)] # One sentence for simplicity

        return DummyStanzaDoc(text)


# ============================================================================
# GŁÓWNY PROCESOR GTMØ
# ============================================================================

class GTMOProcessor:
    """Główny procesor analizy GTMØ."""

    def __init__(self):
        print("🔧 Inicjalizuję procesor GTMØ...")
        self.morfeusz = MorfeuszAnalyzer()
        self.spacy = SpacyAnalyzer()
        self.stanza = StanzaAnalyzer() # Added Stanza initialization
        self._init_mappings()
        print("✅ Procesor GTMØ gotowy")

    def _init_mappings(self):
        """Inicjalizacja mapowań morfologicznych."""
        self.case_map = {case.tag: case for case in PolishCase}
        # Mapowanie POS tagów spaCy/Stanza na enum PolishPOS
        self.pos_map = {
            'NOUN': PolishPOS.SUBST,
            'ADJ': PolishPOS.ADJ,
            'VERB': PolishPOS.VERB,
            'ADV': PolishPOS.ADV,
            'NUM': PolishPOS.NUM,
            'PRON': PolishPOS.PRON,
            'ADP': PolishPOS.PREP,
            'CCONJ': PolishPOS.CONJ,
            'SCONJ': PolishPOS.CONJ,
            'PART': PolishPOS.PART,
            'PUNCT': PolishPOS.INTERP,
            'DET': PolishPOS.PRON, # Mapping DET to PRON as it's not in PolishPOS enum
            'X': PolishPOS.PART # Mapping X to PART as a general fallback
        }


    def calculate_coordinates(self, text: str) -> Tuple[GTMOCoordinates, Configuration, Dict]:
        """Oblicz współrzędne GTMØ dla tekstu."""
        config = Configuration()
        metadata = {'text': text, 'analyses': [], 'morphological_features': []}

        # Ternary Stream 1: Morphological analysis (Morfeusz or fallback)
        morfeusz_analysis = self.morfeusz.analyze(text)
        morph_coords = self._process_morphology(morfeusz_analysis, metadata)

        # Ternary Stream 2: Syntactic analysis (spaCy or fallback)
        spacy_doc = self.spacy.analyze(text)
        syntax_coords, config = self._process_syntax(spacy_doc, metadata) # Pass spacy doc

        # Ternary Stream 3: Semantic computation (Simplified)
        semantic_coords = self._compute_semantics(text)

        # Combine three streams with weighted average
        weights = [0.4, 0.35, 0.25]  # Morphology, Syntax, Semantics
        combined = np.average(
            [morph_coords.to_array(), syntax_coords.to_array(), semantic_coords.to_array()],
            weights=weights,
            axis=0
        )

        coords = GTMOCoordinates(*combined)
        # config.position and config.scale set in _process_syntax
        config.time = datetime.now().timestamp()


        # Classify to nearest attractor
        metadata['attractor'] = self._find_nearest_attractor(coords)

        return coords, config, metadata

    def _process_morphology(self, morfeusz_analysis: List[Dict], metadata: Dict) -> GTMOCoordinates:
        """Przetwarzanie cech morfologicznych z rozszerzoną analizą."""
        coords_list = []

        for analysis in morfeusz_analysis:
            tag_parts = analysis.get('tag', '').split(':')
            tag = analysis.get('tag', '')
            form = analysis.get('form', '')
            lemma = analysis.get('lemma', '')

            # ISTNIEJĄCE: Wyciągnij przypadek
            for tag_part in tag_parts:
                if tag_part in self.case_map:
                    case = self.case_map[tag_part]
                    coords_list.append(case.coords.to_array())
                    metadata['analyses'].append({
                        'type': 'case',
                        'value': case.polish_name,
                        'source': analysis.get('source', 'morfeusz2' if self.morfeusz.available else 'fallback')
                    })
                    break

            # NOWE: Wykryj i dodaj rodzaj gramatyczny
            if 'subst' in tag or 'adj' in tag:
                gender = self.morfeusz._detect_gender(form, tag)
                if gender in GENDER_COORDS:
                    coords_list.append(np.array(GENDER_COORDS[gender]))
                    metadata['morphological_features'].append({
                        'type': 'gender',
                        'value': gender,
                        'form': form
                    })

            # NOWE: Wykryj i dodaj liczbę
            number = self.morfeusz._detect_number(form)
            if number in NUMBER_COORDS:
                coords_list.append(np.array(NUMBER_COORDS[number]))
                metadata['morphological_features'].append({
                    'type': 'number',
                    'value': number,
                    'form': form
                })

            # NOWE: Dla czasowników - aspekt i czas
            if 'verb' in tag:
                # Aspekt
                aspect = self.morfeusz._detect_verb_aspect(lemma)
                if aspect in ASPECT_COORDS:
                    coords_list.append(np.array(ASPECT_COORDS[aspect]))
                    metadata['morphological_features'].append({
                        'type': 'aspect',
                        'value': aspect,
                        'lemma': lemma
                    })

                # Czas
                tense = self.morfeusz._detect_verb_tense(form)
                if tense in TENSE_COORDS:
                    coords_list.append(np.array(TENSE_COORDS[tense]))
                    metadata['morphological_features'].append({
                        'type': 'tense',
                        'value': tense,
                        'form': form
                    })

            # NOWE: Dla przymiotników - stopień
            if 'adj' in tag:
                degree = self.morfeusz._detect_adj_degree(form)
                if degree in DEGREE_COORDS:
                    coords_list.append(np.array(DEGREE_COORDS[degree]))
                    metadata['morphological_features'].append({
                        'type': 'degree',
                        'value': degree,
                        'form': form
                    })

        if coords_list:
            return GTMOCoordinates(*np.mean(coords_list, axis=0))
        return GTMOCoordinates(0.5, 0.5, 0.5)

    def _process_syntax(self, spacy_doc: Optional[Any], metadata: Dict) -> Tuple[GTMOCoordinates, Configuration]:
        """Przetwarzanie cech syntaktycznych."""
        coords_list = []
        config = Configuration() # Initialize config here

        if spacy_doc:
            try:
                # Analyze dependency tree structure
                dependency_scores = {
                    'ROOT': (0.9, 0.8, 0.1),      # Root verb - high determination
                    'nsubj': (0.8, 0.7, 0.2),     # Nominal subject
                    'obj': (0.7, 0.6, 0.3),       # Direct object
                    'iobj': (0.6, 0.5, 0.4),      # Indirect object
                    'ccomp': (0.5, 0.4, 0.5),     # Clausal complement
                    'xcomp': (0.5, 0.4, 0.5),     # Open clausal complement
                    'advcl': (0.4, 0.3, 0.6),     # Adverbial clause
                    'advmod': (0.3, 0.4, 0.6),    # Adverbial modifier
                    'amod': (0.6, 0.5, 0.4),      # Adjectival modifier
                    'nmod': (0.5, 0.6, 0.4),      # Nominal modifier
                    'det': (0.8, 0.8, 0.2),       # Determiner
                    'case': (0.7, 0.7, 0.3),      # Case marking
                    'punct': (0.9, 0.9, 0.1),     # Punctuation
                    'conj': (0.5, 0.5, 0.5),      # Conjunction
                    'cc': (0.5, 0.5, 0.5),        # Coordinating conjunction
                    'aux': (0.6, 0.6, 0.4),       # Auxiliary
                    'cop': (0.7, 0.7, 0.3),       # Copula
                    'mark': (0.4, 0.4, 0.6),      # Marker
                    'appos': (0.5, 0.4, 0.5),     # Appositional modifier
                    'acl': (0.4, 0.3, 0.6),       # Adjectival clause
                    'dep': (0.3, 0.3, 0.7)        # Unspecified dependency
                }

                root_count = 0
                total_depth = 0
                max_depth = 0

                for token in spacy_doc:
                    # Map POS tags
                    if token.pos_ in self.pos_map:
                        pos_enum = self.pos_map[token.pos_]
                        coords_list.append(pos_enum.coords.to_array())
                        metadata['analyses'].append({
                            'type': 'pos',
                            'value': pos_enum.polish_name, # Correctly access polish_name from enum member
                            'token': token.text,
                            'source': 'spacy' if self.spacy.available else 'fallback'
                        })
                    elif token.pos_ != 'SPACE': # Ignore whitespace tokens
                         metadata['analyses'].append({
                            'type': 'pos',
                            'value': token.pos_, # Use raw POS if not in map
                            'token': token.text,
                            'source': 'spacy' if self.spacy.available else 'fallback'
                        })


                    # Analyze dependency relations
                    if token.dep_ in dependency_scores:
                        dep_coords = dependency_scores[token.dep_]
                        coords_list.append(np.array(dep_coords))
                        metadata['analyses'].append({
                            'type': 'dependency',
                            'value': token.dep_,
                            'token': token.text,
                            'source': 'spacy' if self.spacy.available else 'fallback'
                        })
                    elif token.dep_ != 'punct': # Ignore punctuation dependencies
                         metadata['analyses'].append({
                            'type': 'dependency',
                            'value': token.dep_,
                            'token': token.text,
                            'source': 'spacy' if self.spacy.available else 'fallback'
                        })


                    # Calculate syntactic tree depth
                    # Check if token has head before accessing ancestors
                    if token.head != token:
                        ancestors = list(token.ancestors)
                        depth = len(ancestors)
                        total_depth += depth
                        max_depth = max(max_depth, depth)

                    # Special handling for ROOT
                    if token.dep_ == "ROOT":
                        root_count += 1
                        config.position[0] = 0.8  # High semantic centrality

                        # Analyze verb properties for Polish
                        if token.pos_ == "VERB":
                            # Safely access morph attributes and handle lists
                            aspect_data = None
                            mood_data = None
                            if hasattr(token, 'morph') and token.morph is not None:
                                 # Use .get() to retrieve the list from the morph object
                                 if hasattr(token.morph, 'get'): # spaCy Morph or similar with .get
                                     aspect_data = token.morph.get("Aspect")
                                     mood_data = token.morph.get("Mood")
                                 elif isinstance(token.morph, dict): # Fallback case
                                     aspect_data = token.morph.get("Aspect")
                                     mood_data = token.morph.get("Mood")


                            if aspect_data and isinstance(aspect_data, list) and len(aspect_data) > 0: # Check if list and not empty
                                aspect = str(aspect_data[0]) # Access element by index
                                if aspect == "Perf":
                                    config.position[1] = 0.7  # Perfective - more determined
                                else:
                                    config.position[1] = 0.4  # Imperfective - less determined

                            if mood_data and isinstance(mood_data, list) and len(mood_data) > 0: # Check if list and not empty
                                mood = str(mood_data[0]) # Access element by index
                                mood_values = {
                                    "Ind": 0.8,  # Indicative - factual
                                    "Imp": 0.6,  # Imperative - directive
                                    "Cnd": 0.3   # Conditional - hypothetical
                                }
                                config.position[2] = mood_values.get(mood, 0.5)


                    # Analyze Polish-specific features (integrated into the main loop)
                    if token.pos_ == "ADP":  # Preposition
                        # Different prepositions require different cases
                        prep_case_patterns = {
                            'w': (0.6, 0.7, 0.3),    # w + locative
                            'na': (0.7, 0.6, 0.3),   # na + accusative/locative
                            'z': (0.5, 0.5, 0.5),    # z + genitive/instrumental
                            'do': (0.8, 0.7, 0.2),   # do + genitive
                            'od': (0.7, 0.6, 0.3),   # od + genitive
                            'przez': (0.6, 0.5, 0.4), # przez + accusative
                            'dla': (0.6, 0.6, 0.4),  # dla + genitive
                            'o': (0.5, 0.4, 0.5),    # o + locative/accusative
                            'po': (0.4, 0.3, 0.6),   # po + locative/accusative
                            'za': (0.5, 0.5, 0.5)    # za + instrumental/accusative
                        }
                        if token.text.lower() in prep_case_patterns:
                            coords_list.append(np.array(prep_case_patterns[token.text.lower()]))


                # Calculate average tree depth for abstraction scale
                avg_depth = total_depth / len(spacy_doc) if len(spacy_doc) > 0 else 1
                config.scale = min(1.0, avg_depth / 5.0)

                # Analyze sentence complexity
                if len(spacy_doc) > 0:
                    # Simple sentence vs complex sentence
                    if root_count == 1 and max_depth <= 2:
                        # Simple sentence - higher stability
                        coords_list.append(np.array([0.7, 0.8, 0.2]))
                    elif root_count > 1 or max_depth > 4:
                        # Complex sentence - higher entropy
                        coords_list.append(np.array([0.4, 0.3, 0.7]))
                    else:
                        # Medium complexity
                        coords_list.append(np.array([0.5, 0.5, 0.5]))


            except Exception as e:
                logger.warning(f"Błąd przetwarzania spaCy: {e}")

        if coords_list:
            coords = GTMOCoordinates(*np.mean(coords_list, axis=0))
        else:
            coords = GTMOCoordinates(0.5, 0.5, 0.5)

        return coords, config

    def _compute_semantics(self, text: str) -> GTMOCoordinates:
        """Oblicz cechy semantyczne."""
        # Simplified semantic analysis
        text_lower = text.lower()

        determination = 0.5
        stability = 0.5
        entropy = 0.5

        # Count sentence type indicators
        if '?' in text:
            entropy += 0.3
            stability -= 0.2
            determination -= 0.1
        elif '!' in text:
            determination += 0.1
            entropy += 0.1
        elif '.' in text:
            stability += 0.1
            determination += 0.05

        # Simple sentiment check (very basic)
        positive_words = ['dobry', 'miły', 'szczęście', 'radość', 'piękny']
        negative_words = ['zły', 'smutny', 'ból', 'strach', 'śmierć']

        sentiment_score = 0
        for word in text_lower.split():
            if word in positive_words:
                sentiment_score += 1
            elif word in negative_words:
                sentiment_score -= 1

        if sentiment_score > 0:
            determination += 0.1 # Positive sentiment adds some clarity
        elif sentiment_score < 0:
            entropy += 0.1 # Negative sentiment might add complexity

        # Clip to valid range
        coords = GTMOCoordinates(
            np.clip(determination, 0, 1),
            np.clip(stability, 0, 1),
            np.clip(entropy, 0, 1)
        )

        return coords


    def _find_nearest_attractor(self, coords: GTMOCoordinates) -> Dict:
        """Znajdź najbliższy atraktor wiedzy."""
        min_distance = float('inf')
        nearest = None

        for attractor in KnowledgeAttractor:
            distance = coords.distance_to(attractor.coords)
            if distance < min_distance:
                min_distance = distance
                nearest = attractor

        return {
            'symbol': nearest.symbol,
            'name': nearest.display_name,
            'distance': float(min_distance),
            'coordinates': {
                'determination': nearest.coords.determination,
                'stability': nearest.coords.stability,
                'entropy': nearest.coords.entropy
            }
        }

# ============================================================================
# WIZUALIZATOR Z FALLBACK
# ============================================================================

class GTMOVisualizer:
    """Silnik wizualizacji wyników analizy GTMØ."""

    def __init__(self):
        self.available = True # Assume Plotly is available after installation attempt
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
        except ImportError:
            self.available = False
            logger.warning("Plotly niedostępny - wizualizacje wyłączone")


        self.attractor_colors = {
            'Ψᴷ': '#FF6B6B',  # Red
            'Ψʰ': '#4ECDC4',  # Cyan
            'Ψᴺ': '#45B7D1',  # Blue
            'Ø': '#96CEB4',   # Green
            'Ψ~': '#FECA57',  # Yellow
            'Ψ◊': '#DDA0DD'   # Plum
        }

    def create_configuration_space_plot(self, results: List[Dict]):
        """Utwórz interaktywną wizualizację 3D Przestrzeni Konfiguracyjnej."""
        if not self.available:
            print("❌ Plotly niedostępny - nie można utworzyć wizualizacji")
            return None

        try:
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{'type': 'scatter3d', 'rowspan': 2}, {'type': 'scatter'}],
                    [None, {'type': 'scatter'}]
                ],
                subplot_titles=(
                    'Przestrzeń Konfiguracyjna (3D)',
                    'Determinacja vs Stabilność',
                    'Rozkład Entropii'
                )
            )

            # Wyciągnij dane
            sentences = [r['text'] for r in results]
            coords = [r['coordinates'] for r in results]
            attractors = [r['metadata']['attractor'] for r in results]

            # Wykres 3D scatter
            x = [c['determination'] for c in coords]
            y = [c['stability'] for c in coords]
            z = [c['entropy'] for c in coords]
            colors = [self.attractor_colors.get(a['symbol'], '#808080') for a in attractors]

            # Dodaj punkty zdań
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color=colors,
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"{i+1}" for i in range(len(sentences))],
                    textposition='top center',
                    hovertemplate='<b>Zdanie %{text}</b><br>' +
                                  'Determinacja: %{x:.3f}<br>' +
                                  'Stabilność: %{y:.3f}<br>' +
                                  'Entropia: %{z:.3f}<br>' +
                                  '<extra></extra>',
                    name='Zdania'
                ),
                row=1, col=1
            )

            # Dodaj atraktory
            for attractor in KnowledgeAttractor:
                fig.add_trace(
                    go.Scatter3d(
                        x=[attractor.coords.determination],
                        y=[attractor.coords.stability],
                        z=[attractor.coords.entropy],
                        mode='markers+text',
                        marker=dict(
                            size=20,
                            color=self.attractor_colors.get(attractor.symbol, '#808080'),
                            symbol='diamond',
                            opacity=1.0,
                            line=dict(width=2, color='black')
                        ),
                        text=[attractor.symbol],
                        textposition='top center',
                        name=attractor.display_name,
                        hovertemplate=f'<b>{attractor.display_name}</b><br>' +
                                      'D: %{x:.3f}<br>S: %{y:.3f}<br>E: %{z:.3f}<br>' +
                                      '<extra></extra>'
                    ),
                    row=1, col=1
                )

            # Projekcja 2D: Determinacja vs Stabilność
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Entropia', x=1.15)
                    ),
                    text=[f"S{i+1}" for i in range(len(sentences))],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'D: %{x:.3f}<br>S: %{y:.3f}<br>' +
                                  '<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )

            # Rozkład entropii
            fig.add_trace(
                go.Histogram(
                    x=z,
                    nbinsx=20,
                    marker_color='#45B7D1',
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=2
            )

            # Actual subplot axis titles based on make_subplots structure
            fig.update_layout(
                title={
                    'text': 'Analiza Przestrzeni Konfiguracyjnej GTMØ',
                    'font': {'size': 20, 'color': '#2C3E50'}
                },
                showlegend=True,
                height=800,
                scene=dict(  # This is for the 3D plot (row 1, col 1)
                    xaxis_title='Determinacja →',
                    yaxis_title='Stabilność →',
                    zaxis_title='Entropia →',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                xaxis2_title='Determinacja', # Correct axis reference for row 1, col 2
                yaxis2_title='Stabilność',   # Correct axis reference for row 1, col 2
                xaxis3=dict(title='Entropia'), # Correct way to set title for axis 3
                yaxis3=dict(title='Liczba')   # Correct way to set title for axis 3
            )

            return fig

        except Exception as e:
            logger.error(f"Błąd tworzenia wizualizacji: {e}")
            return None

    def show_text_summary(self, results: List[Dict]):
        """Pokaż podsumowanie tekstowe gdy wizualizacje niedostępne."""
        print("\n" + "="*60)
        print("PODSUMOWANIE ANALIZY GTMØ")
        print("="*60)

        for i, result in enumerate(results, 1):
            coords = result['coordinates']
            attractor = result['metadata']['attractor']

            print(f"\n{i}. '{result['text']}'")
            print(f"   📊 Współrzędne: [{coords['determination']:.3f}, {coords['stability']:.3f}, {coords['entropy']:.3f}]")
            print(f"   🎯 Atraktor: {attractor['symbol']} ({attractor['name']})")
            print(f"   📏 Dystans: {attractor['distance']:.3f}")

        # Statystyki ogólne
        if results:
            avg_det = np.mean([r['coordinates']['determination'] for r in results])
            avg_stab = np.mean([r['coordinates']['stability'] for r in results])
            avg_ent = np.mean([r['coordinates']['entropy'] for r in results])

            print(f"\n📈 STATYSTYKI ŚREDNIE:")
            print(f"   Determinacja: {avg_det:.3f}")
            print(f"   Stabilność: {avg_stab:.3f}")
            print(f"   Entropia: {avg_ent:.3f}")
        else:
            print("\n📈 Brak wyników do podsumowania.")

# ============================================================================
# GŁÓWNA KLASA ANALIZATORA
# ============================================================================

class GTMOAnalyzer:
    """Główny analizator orkiestrujący kompletny pipeline analizy GTMØ."""
    def __init__(self):
        global installation_status

        logger.info("Inicjalizuję Analizator GTMØ...")

        # Check if installation status is available and indicates success
        # Używamy `globals()` do bezpiecznego sprawdzenia istnienia zmiennej
        if 'installation_status' not in globals() or not installation_status.get('morfeusz') or not installation_status.get('spacy') or not installation_status.get('stanza'):
            logger.warning("⚠️  Zależności nie zostały pomyślnie zainstalowane. Analiza może używać fallbacków.")

            # Próba ponownej instalacji
            installation_status = install_dependencies()

        self.processor = GTMOProcessor()
        self.visualizer = GTMOVisualizer()
        self.test_sentences = self._get_test_sentences()


    def _get_test_sentences(self) -> List[str]:
        """Pobierz 15 różnorodnych polskich zdań testowych."""
        return [
            # Fakty i pewności
            "Warszawa jest stolicą Polski.",
            "Dwa plus dwa równa się cztery.",

            # Pytania
            "Czy sprawiedliwość zawsze zwycięża?",
            "Kiedy nadejdzie prawdziwa wolność?",

            # Niepewność
            "Może jutro będzie padać deszcz.",
            "Chyba nie zdążymy na pociąg.",

            # Pojęcia abstrakcyjne
            "Miłość jest silniejsza niż śmierć.",
            "Sprawiedliwość powinna być ślepa.",

            # Paradoksy
            "To zdanie jest fałszywe.",
            "Wiem, że nic nie wiem.",

            # Polecenia
            "Przyjdź jutro o ósmej rano!",
            "Nie zapomnij o spotkaniu.",

            # Emocjonalne
            "Och, jak pięknie pachną te róże!",

            # Złożone
            "Choć burza szaleje, statek płynie dalej.",
            "Gdyby nie deszcz, poszlibyśmy na spacer."
        ]

    def analyze_all(self) -> List[Dict]:
        """Analizuj wszystkie zdania testowe."""
        results = []

        for i, sentence in enumerate(self.test_sentences, 1):
            logger.info(f"Analizuję zdanie {i}/{len(self.test_sentences)}: {sentence[:50]}...")

            try:
                coords, config, metadata = self.processor.calculate_coordinates(sentence)

                result = {
                    'id': i,
                    'text': sentence,
                    'coordinates': {
                        'determination': float(coords.determination),
                        'stability': float(coords.stability),
                        'entropy': float(coords.entropy)
                    },
                    'configuration': config.to_dict(),
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                }

                results.append(result)

            except Exception as e:
                logger.error(f"❌ Błąd podczas analizy zdania '{sentence[:50]}...': {e}")
                # Optionally append a failed analysis marker
                results.append({
                    'id': i,
                    'text': sentence,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                continue


        return results

    def save_results(self, results: List[Dict]) -> str:
        """Zapisz wyniki do pliku JSON z timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'gtmo_analysis_{timestamp}.json'

        output = {
            'version': '2.1',
            'timestamp': datetime.now().isoformat(),
            'total_sentences': len([r for r in results if r.get('status') != 'failed']),
            'failed_sentences': len([r for r in results if r.get('status') == 'failed']),
            'installation_status': installation_status,
            'results': results,
            'statistics': self._calculate_statistics(results)
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Wyniki zapisane do {filename}")
        return filename

    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Oblicz statystyki podsumowujące (tylko dla udanych analiz)."""
        successful_results = [r for r in results if r.get('status') != 'failed']

        if not successful_results:
             return {}

        coords = [r['coordinates'] for r in successful_results]

        return {
            'mean_determination': float(np.mean([c['determination'] for c in coords])),
            'mean_stability': float(np.mean([c['stability'] for c in coords])),
            'mean_entropy': float(np.mean([c['entropy'] for c in coords])),
            'std_determination': float(np.std([c['determination'] for c in coords])),
            'std_stability': float(np.std([c['stability'] for c in coords])),
            'std_entropy': float(np.std([c['entropy'] for c in coords])),
            'attractor_distribution': self._count_attractors(successful_results)
        }

    def _count_attractors(self, results: List[Dict]) -> Dict:
        """Policz rozkład najbliższych atraktorów (tylko dla udanych analiz)."""
        attractors = {}
        for result in results:
            if 'metadata' in result and 'attractor' in result['metadata']:
                 symbol = result['metadata']['attractor']['symbol']
                 attractors[symbol] = attractors.get(symbol, 0) + 1
        return attractors


    def run_complete_analysis(self):
        """Uruchom kompletny pipeline analizy."""
        print("\n" + "="*60)
        print("ANALIZATOR GTMØ POLISH NLP")
        print("="*60 + "\n")

        # Analizuj zdania
        print("🔍 Analizuję zdania...")
        results = self.analyze_all()

        # Zapisz wyniki
        if results:
            print("\n💾 Zapisuję wyniki...")
            filename = self.save_results(results)
            print(f"✅ Wyniki zapisane do: {filename}")

            # Utwórz wizualizacje (tylko dla udanych analiz)
            successful_results = [r for r in results if r.get('status') != 'failed']
            if successful_results:
                print("\n📊 Generuję wizualizacje...")

                if self.visualizer.available:
                    fig1 = self.visualizer.create_configuration_space_plot(successful_results)

                    if fig1:
                        html_filename = 'gtmo_configuration_space.html'
                        fig1.write_html(html_filename)
                        print(f"✅ Wizualizacja zapisana do {html_filename}")
                        fig1.show()
                    else:
                        print("❌ Błąd generowania wizualizacji")
                        self.visualizer.show_text_summary(successful_results)
                else:
                    print("ℹ️ Plotly niedostępny - pokazuję podsumowanie tekstowe")
                    self.visualizer.show_text_summary(successful_results)
            else:
                 print("\nℹ️ Brak udanych analiz do wizualizacji.")

        else:
             print("\n❌ Brak wyników analizy do zapisania lub wizualizacji.")


        print("\n" + "="*60)
        print("ANALIZA ZAKOŃCZONA")
        print("="*60)

        return results

# ============================================================================
# WYKONANIE
# ============================================================================

if __name__ == "__main__":
    # Inicjalizuj i uruchom analizator
    try:
        analyzer = GTMOAnalyzer()
        results = analyzer.run_complete_analysis()

        print(f"\n✅ GTMØ Analiza zakończona!")
        print(f"Przetworzono {len(results)} zdań ({len([r for r in results if r.get('status') != 'failed'])} udanych)")
        print("Sprawdź wygenerowane pliki dla szczegółowych wyników.")

    except Exception as e:
        print(f"\n❌ Krytyczny błąd podczas inicjalizacji lub uruchamiania: {e}")
        import traceback
        traceback.print_exc()
