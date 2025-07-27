Dokumentacja Testów dla Logiki SANB-5
Ten dokument opisuje zestaw testów jednostkowych i wydajnościowych zaimplementowanych w pliku test_sanb5_logic.py. Celem tych testów jest weryfikacja poprawności, spójności i wydajności implementacji 5-wartościowej logiki SANB-5 oraz jej integracji z koncepcjami teorii GTMØ.

1. Testy Jednostkowe (unittest)
Testy jednostkowe mają na celu weryfikację poprawności działania poszczególnych komponentów systemu w izolacji. Zostały one podzielone na cztery główne klasy, z których każda odpowiada za inny moduł logiki.
TestPhaseSpaceCoordinates
Ta klasa testowa sprawdza, czy mapowanie wartości logicznych SANB-5 na współrzędne w 3-wymiarowej przestrzeni fazowej GTMØ jest prawidłowe.
test_from_sanb5_mapping:
Cel: Weryfikacja, czy każda z pięciu wartości logicznych (O, Z, PHI, INF, PSI) jest poprawnie konwertowana na przypisany jej zestaw współrzędnych (Determinizm, Stabilność, Entropia).
Metoda: Test iteruje po słowniku oczekiwanych mapowań i sprawdza, czy dla każdej wartości logicznej funkcja from_sanb5 zwraca prawidłowe współrzędne z odpowiednią precyzją.
TestSANB5Logic
Jest to najbardziej rozbudowana klasa testowa, która weryfikuje działanie wszystkich fundamentalnych operatorów logicznych SANB-5.
test_negation:
Cel: Sprawdzenie poprawności działania operatora negacji (¬).
Metoda: Weryfikuje, czy O jest negowane do Z (i odwrotnie), a stany niedookreślone (PHI, INF, PSI) pozostają niezmienione.
test_all_binary_operators_completeness:
Cel: Zapewnienie, że wszystkie operatory binarne (AND, OR, XOR, etc.) mają zdefiniowane wyniki dla każdej możliwej kombinacji 5x5 wartości wejściowych.
Metoda: Sprawdza, czy każda tablica prawdy zawiera dokładnie 25 wpisów.
test_conjunction (AND):
Cel: Weryfikacja logiki koniunkcji, działającej na zasadzie "najsłabszego ogniwa".
Metoda: Sprawdza kluczowe przypadki, np. czy Z (fałsz) "pochłania" każdą inną wartość, a PHI (osobliwość) pochłania wszystko oprócz Z.
test_disjunction (OR):
Cel: Weryfikacja logiki dysjunkcji, działającej na zasadzie "najsilniejszego ogniwa".
Metoda: Sprawdza, czy O (prawda) dominuje nad każdą inną wartością, a INF (chaos) dominuje nad wszystkim oprócz O.
test_implication:
Cel: Sprawdzenie operatora implikacji, modelującego "przepływ pewności".
Metoda: Testuje reguły, takie jak "z fałszu wynika wszystko" (Z → x = O) oraz "z prawdy wynika następnik" (O → x = x).
test_equivalence:
Cel: Weryfikacja operatora równoważności, który testuje tożsamość ontologiczną.
Metoda: Sprawdza, czy identyczne stany (np. PSI ↔ PSI) dają w wyniku prawdę (O), a różne stany dają wyniki zgodne z tablicą prawdy.
test_xor:
Cel: Weryfikacja operatora XOR, czyli testu rygorystycznej różnicy.
Metoda: Sprawdza, czy O ⊕ Z daje O (jak w klasycznej logice) oraz jak operator radzi sobie ze stanami niedookreślonymi. Zaznaczony przez Ciebie fragment kodu elif SANB5Value.PHI in (v1, v2): jest częścią logiki tego operatora i zapewnia, że jeśli którakolwiek z wartości wejściowych to PHI, wynik również jest PHI. Ten test weryfikuje m.in. tę regułę.
TestGTMOTransformations
Ta klasa weryfikuje funkcje, które modelują specyficzne koncepcje z teorii GTMØ.
test_question_transformation:
Cel: Sprawdzenie, czy transformacja "pytająca" poprawnie zmienia stany definitywne (O, Z) w superpozycję (PSI), pozostawiając stany niedookreślone bez zmian.
test_observation_collapse:
Cel: Weryfikacja modelu kolapsu superpozycji (PSI) w wyniku obserwacji.
Metoda: Sprawdza, czy w zależności od wartości observer_bias, stan PSI poprawnie zapada się do O, Z, PHI, INF lub pozostaje w superpozycji.
test_alienated_number_operation:
Cel: Sprawdzenie reguły, zgodnie z którą operacje matematyczne na bytach niedookreślonych (reprezentowanych przez PHI, INF, PSI) prowadzą do kolapsu do osobliwości (PHI).
TestInterpretationEngine
Ta klasa testuje mechanizm do zarządzania i analizowania wielu, często sprzecznych, interpretacji.
test_add_interpretation:
Cel: Prosty test sprawdzający, czy interpretacje są poprawnie dodawane do silnika.
test_analyze_consistency:
Cel: Weryfikacja logiki analizującej spójność zbioru interpretacji.
Metoda: Sprawdza różne scenariusze: brak interpretacji (wynik PHI), zbiór spójny, zbiór sprzeczny (O i Z, wynik PHI), dominacja chaosu (INF) oraz inne mieszanki (wynik PSI).
test_trajectory_productivity:
Cel: Sprawdzenie, czy funkcja poprawnie ocenia trajektorię zmian interpretacji jako produktywną, nieproduktywną lub neutralną na podstawie ostatnich trzech kroków.

2. Testy Wydajnościowe (timeit)
Testy te, uruchamiane po pomyślnym przejściu testów jednostkowych, mierzą szybkość wykonania kluczowych operacji.
Cel: Ocena, czy implementacja jest zoptymalizowana i nie zawiera "wąskich gardeł", które mogłyby spowalniać system przy intensywnym użytkowaniu.
Metoda: Moduł timeit wykonuje każdą z testowanych operacji w pętli (np. milion razy) i mierzy łączny czas.
Mierzone operacje:
Operacja logiczna (AND): Reprezentuje wydajność podstawowych, najczęściej wykonywanych operacji logicznych, które opierają się na odczycie z tablicy prawdy.
Transformacja GTMØ (Question): Mierzy wydajność prostych transformacji, które również są kluczowe dla działania systemu.
Analiza spójności (InterpretationEngine): Mierzy wydajność bardziej złożonej operacji, która wymaga iteracji i analizy listy, co pozwala ocenić wydajność systemu przy bardziej skomplikowanych zadaniach.
