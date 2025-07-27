# test_sanb5_logic.py (Combined and Fixed)
# This file contains the SANB-5 logic code and comprehensive unit and performance tests.
# Combining both files into one resolves the 'ModuleNotFoundError'.

import unittest
import timeit
import itertools
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, List

# --- SECTION 1: SANB-5 LOGIC CODE (from gtmo_sanb5_logic.py) ---

class SANB5Value(Enum):
    """The five logical values in the SANB-5 system."""
    O = "O"  # One/True - Definite truth
    Z = "Z"  # Zero/False - Definite falsehood
    PHI = "Ø"  # Singularity - Boundary of definition
    INF = "∞"  # Infinity/Chaos - Maximum entropy
    PSI = "Ψ"  # Superposition - Quantum state
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"SANB5.{self.name}"

@dataclass
class PhaseSpaceCoordinates:
    """Maps SANB-5 values to GTMØ Phase Space coordinates."""
    determinacy: float
    stability: float
    entropy: float
    
    @classmethod
    def from_sanb5(cls, value: SANB5Value) -> 'PhaseSpaceCoordinates':
        """Converts an SANB-5 value to Phase Space coordinates."""
        mapping = {
            SANB5Value.O: cls(0.95, 0.95, 0.05),
            SANB5Value.Z: cls(0.95, 0.95, 0.05),
            SANB5Value.PHI: cls(0.01, 0.01, 0.01),
            SANB5Value.INF: cls(0.05, 0.05, 0.95),
            SANB5Value.PSI: cls(0.50, 0.30, 0.70),
        }
        return mapping[value]

class SANB5Logic:
    """Implementation of SANB-5 logical operators."""
    
    def __init__(self):
        """Initializes the SANB-5 logic system with truth tables."""
        self._init_truth_tables()
        
    def _init_truth_tables(self):
        """Initializes all truth tables for SANB-5 operators."""
        self.negation_table = {
            SANB5Value.O: SANB5Value.Z,
            SANB5Value.Z: SANB5Value.O,
            SANB5Value.PHI: SANB5Value.PHI,
            SANB5Value.INF: SANB5Value.INF,
            SANB5Value.PSI: SANB5Value.PSI,
        }
        self.conjunction_table = self._create_conjunction_table()
        self.disjunction_table = self._create_disjunction_table()
        self.implication_table = self._create_implication_table()
        self.equivalence_table = self._create_equivalence_table()
        self.xor_table = self._create_xor_table()
    
    def _create_conjunction_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        table = {}
        values = list(SANB5Value)
        for v1, v2 in itertools.product(values, repeat=2):
            if v1 == SANB5Value.Z or v2 == SANB5Value.Z: table[(v1, v2)] = SANB5Value.Z
            elif v1 == SANB5Value.PHI or v2 == SANB5Value.PHI: table[(v1, v2)] = SANB5Value.PHI
            elif v1 == v2: table[(v1, v2)] = v1
            elif SANB5Value.INF in (v1, v2): table[(v1, v2)] = SANB5Value.INF
            else: table[(v1, v2)] = SANB5Value.PSI
        return table
    
    def _create_disjunction_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        table = {}
        values = list(SANB5Value)
        for v1, v2 in itertools.product(values, repeat=2):
            if v1 == SANB5Value.O or v2 == SANB5Value.O: table[(v1, v2)] = SANB5Value.O
            elif v1 == SANB5Value.INF or v2 == SANB5Value.INF: table[(v1, v2)] = SANB5Value.INF
            elif v1 == v2: table[(v1, v2)] = v1
            elif SANB5Value.PSI in (v1, v2): table[(v1, v2)] = SANB5Value.PSI
            else: table[(v1, v2)] = SANB5Value.PHI
        return table
    
    def _create_implication_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        table = {}
        values = list(SANB5Value)
        for v1, v2 in itertools.product(values, repeat=2):
            if v1 == SANB5Value.Z: table[(v1, v2)] = SANB5Value.O
            elif v1 == SANB5Value.O: table[(v1, v2)] = v2
            elif v1 == SANB5Value.PHI: table[(v1, v2)] = SANB5Value.O if v2 == SANB5Value.O else SANB5Value.PHI
            elif v1 == SANB5Value.INF: table[(v1, v2)] = SANB5Value.O if v2 == SANB5Value.O else SANB5Value.INF
            elif v1 == SANB5Value.PSI: table[(v1, v2)] = SANB5Value.O if v2 == SANB5Value.O else SANB5Value.PSI
        return table
    
    def _create_equivalence_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        table = {}
        values = list(SANB5Value)
        for v1, v2 in itertools.product(values, repeat=2):
            if v1 == v2:
                if v1 in (SANB5Value.O, SANB5Value.Z, SANB5Value.PSI): table[(v1, v2)] = SANB5Value.O
                else: table[(v1, v2)] = v1
            else:
                if {v1, v2} == {SANB5Value.O, SANB5Value.Z}: table[(v1, v2)] = SANB5Value.Z
                elif SANB5Value.PHI in (v1, v2): table[(v1, v2)] = SANB5Value.PHI
                elif SANB5Value.INF in (v1, v2): table[(v1, v2)] = SANB5Value.INF
                else: table[(v1, v2)] = SANB5Value.PSI
        return table
    
    def _create_xor_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        table = {}
        values = list(SANB5Value)
        for v1, v2 in itertools.product(values, repeat=2):
            if v1 == v2:
                if v1 in (SANB5Value.O, SANB5Value.Z, SANB5Value.PSI): table[(v1, v2)] = SANB5Value.Z
                else: table[(v1, v2)] = v1
            else:
                if {v1, v2} == {SANB5Value.O, SANB5Value.Z}: table[(v1, v2)] = SANB5Value.O
                elif SANB5Value.PHI in (v1, v2): table[(v1, v2)] = SANB5Value.PHI
                elif SANB5Value.INF in (v1, v2): table[(v1, v2)] = SANB5Value.INF
                elif SANB5Value.O in (v1, v2): table[(v1, v2)] = SANB5Value.O
                else: table[(v1, v2)] = SANB5Value.PSI
        return table
    
    def neg(self, a: SANB5Value) -> SANB5Value: return self.negation_table[a]
    def and_(self, a: SANB5Value, b: SANB5Value) -> SANB5Value: return self.conjunction_table[(a, b)]
    def or_(self, a: SANB5Value, b: SANB5Value) -> SANB5Value: return self.disjunction_table[(a, b)]
    def implies(self, a: SANB5Value, b: SANB5Value) -> SANB5Value: return self.implication_table[(a, b)]
    def equiv(self, a: SANB5Value, b: SANB5Value) -> SANB5Value: return self.equivalence_table[(a, b)]
    def xor(self, a: SANB5Value, b: SANB5Value) -> SANB5Value: return self.xor_table[(a, b)]

class GTMOTransformations:
    @staticmethod
    def question_transformation(value: SANB5Value) -> SANB5Value:
        return SANB5Value.PSI if value in (SANB5Value.O, SANB5Value.Z) else value
    
    @staticmethod
    def observation_collapse(value: SANB5Value, observer_bias: float = 0.5) -> SANB5Value:
        if value == SANB5Value.PSI:
            if observer_bias < 0.2: return SANB5Value.Z
            elif observer_bias > 0.8: return SANB5Value.O
            elif observer_bias < 0.3: return SANB5Value.PHI
            elif observer_bias > 0.7: return SANB5Value.INF
            else: return SANB5Value.PSI
        return value
    
    @staticmethod
    def alienated_number_operation(v1: SANB5Value, v2: SANB5Value) -> SANB5Value:
        if v1 in (SANB5Value.PHI, SANB5Value.INF, SANB5Value.PSI) or v2 in (SANB5Value.PHI, SANB5Value.INF, SANB5Value.PSI): return SANB5Value.PHI
        elif v1 == SANB5Value.O and v2 == SANB5Value.O: return SANB5Value.O
        else: return SANB5Value.Z

class InterpretationEngine:
    def __init__(self, logic: SANB5Logic):
        self.logic = logic
        self.interpretations: List[Tuple[str, SANB5Value]] = []
    
    def add_interpretation(self, description: str, value: SANB5Value):
        self.interpretations.append((description, value))
    
    def analyze_consistency(self) -> SANB5Value:
        if not self.interpretations: return SANB5Value.PHI
        values = [v for _, v in self.interpretations]
        if len(set(values)) == 1: return values[0]
        if set(values) == {SANB5Value.O, SANB5Value.Z}: return SANB5Value.PHI
        if SANB5Value.INF in values: return SANB5Value.INF
        return SANB5Value.PSI
    
    def trajectory_productivity(self) -> str:
        if len(self.interpretations) < 2: return "Insufficient data"
        recent_values = [v for _, v in self.interpretations[-3:]]
        definite_count = sum(1 for v in recent_values if v in (SANB5Value.O, SANB5Value.Z))
        chaos_count = sum(1 for v in recent_values if v == SANB5Value.INF)
        if definite_count > chaos_count: return "Productive trajectory"
        elif chaos_count > definite_count: return "Unproductive trajectory"
        else: return "Neutral trajectory"


# --- SECTION 2: TEST CODE (from test_sanb5_logic.py) ---

# Shortcuts for logical values for test readability
O, Z, PHI, INF, PSI = (
    SANB5Value.O,
    SANB5Value.Z,
    SANB5Value.PHI,
    SANB5Value.INF,
    SANB5Value.PSI,
)

class TestPhaseSpaceCoordinates(unittest.TestCase):
    """Tests the mapping of SANB-5 values to GTMØ phase space coordinates."""
    def test_from_sanb5_mapping(self):
        """Checks if each SANB-5 value is correctly mapped to its coordinates."""
        test_cases = {
            O: (0.95, 0.95, 0.05), Z: (0.95, 0.95, 0.05),
            PHI: (0.01, 0.01, 0.01), INF: (0.05, 0.05, 0.95),
            PSI: (0.50, 0.30, 0.70),
        }
        for value, expected_coords in test_cases.items():
            with self.subTest(value=value):
                coords = PhaseSpaceCoordinates.from_sanb5(value)
                self.assertAlmostEqual(coords.determinacy, expected_coords[0])
                self.assertAlmostEqual(coords.stability, expected_coords[1])
                self.assertAlmostEqual(coords.entropy, expected_coords[2])

class TestSANB5Logic(unittest.TestCase):
    """Tests the basic logical operators implemented in SANB5Logic."""
    @classmethod
    def setUpClass(cls):
        """Initializes the logic system once for all tests in this class."""
        cls.logic = SANB5Logic()
        cls.all_values = list(SANB5Value)

    def test_negation(self):
        """Tests the negation operator (¬)."""
        expected = {O: Z, Z: O, PHI: PHI, INF: INF, PSI: PSI}
        for val, neg_val in expected.items():
            with self.subTest(value=val):
                self.assertEqual(self.logic.neg(val), neg_val)

    def test_all_binary_operators_completeness(self):
        """Checks if all truth tables for binary operators are fully defined."""
        operators = [
            self.logic.conjunction_table, self.logic.disjunction_table,
            self.logic.implication_table, self.logic.equivalence_table,
            self.logic.xor_table,
        ]
        for op_table in operators:
            with self.subTest(table=op_table):
                # There should be 5x5 = 25 defined pairs
                self.assertEqual(len(op_table), 25)
                for v1, v2 in itertools.product(self.all_values, repeat=2):
                    self.assertIn((v1, v2), op_table)

    def test_conjunction(self):
        """Tests the conjunction operator (∧) - the "weakest link" principle."""
        # A few key cases
        self.assertEqual(self.logic.and_(O, PSI), PSI)
        self.assertEqual(self.logic.and_(Z, INF), Z) # Z absorbs everything
        self.assertEqual(self.logic.and_(PHI, O), PHI) # PHI absorbs everything except Z
        self.assertEqual(self.logic.and_(INF, PSI), INF)

    def test_disjunction(self):
        """Tests the disjunction operator (∨) - the "strongest link" principle."""
        self.assertEqual(self.logic.or_(Z, PSI), PSI)
        self.assertEqual(self.logic.or_(O, INF), O) # O dominates everything
        self.assertEqual(self.logic.or_(PHI, Z), PHI)
        self.assertEqual(self.logic.or_(INF, PSI), INF) # INF dominates everything except O

    def test_implication(self):
        """Tests the implication operator (→) - the flow of certainty."""
        self.assertEqual(self.logic.implies(Z, INF), O) # Anything follows from false
        self.assertEqual(self.logic.implies(O, PSI), PSI) # From true follows the consequent
        self.assertEqual(self.logic.implies(PHI, Z), PHI) # Singularity propagates, unless it implies O

    def test_equivalence(self):
        """Tests the equivalence operator (↔) - the ontological identity test."""
        self.assertEqual(self.logic.equiv(PSI, PSI), O) # Identical states (except PHI/INF) are equivalent
        self.assertEqual(self.logic.equiv(PHI, PHI), PHI) # Equivalence of singularity is singularity
        self.assertEqual(self.logic.equiv(O, Z), Z)
        self.assertEqual(self.logic.equiv(O, PSI), PSI)

    def test_xor(self):
        """Tests the XOR operator (⊕) - the rigorous difference test."""
        self.assertEqual(self.logic.xor(O, Z), O) # Classic XOR
        self.assertEqual(self.logic.xor(PSI, PSI), Z) # Identical states are not different
        self.assertEqual(self.logic.xor(O, PSI), O)
        self.assertEqual(self.logic.xor(Z, PSI), PSI)

class TestGTMOTransformations(unittest.TestCase):
    """Tests the transformations specific to the GTMØ theory."""
    def test_question_transformation(self):
        """Tests the question transformation (0→0?, 1→1?)."""
        self.assertEqual(GTMOTransformations.question_transformation(O), PSI)
        self.assertEqual(GTMOTransformations.question_transformation(Z), PSI)
        self.assertEqual(GTMOTransformations.question_transformation(PHI), PHI) # Indefinite states remain unchanged
        self.assertEqual(GTMOTransformations.question_transformation(INF), INF)
        self.assertEqual(GTMOTransformations.question_transformation(PSI), PSI)

    def test_observation_collapse(self):
        """Tests the collapse of a superposition (Ψ) through observation."""
        # Only PSI should collapse
        self.assertEqual(GTMOTransformations.observation_collapse(O, 0.1), O)
        self.assertEqual(GTMOTransformations.observation_collapse(Z, 0.9), Z)

        # Testing different thresholds for 'observer_bias'
        self.assertEqual(GTMOTransformations.observation_collapse(PSI, 0.1), Z)
        self.assertEqual(GTMOTransformations.observation_collapse(PSI, 0.9), O)
        self.assertEqual(GTMOTransformations.observation_collapse(PSI, 0.25), PHI)
        self.assertEqual(GTMOTransformations.observation_collapse(PSI, 0.75), INF)
        self.assertEqual(GTMOTransformations.observation_collapse(PSI, 0.5), PSI) # Remains in superposition

    def test_alienated_number_operation(self):
        """Tests the collapse to Ø during operations on AlienatedNumbers (indefinite states)."""
        # An operation involving any indefinite state should result in Ø
        self.assertEqual(GTMOTransformations.alienated_number_operation(O, PSI), PHI)
        self.assertEqual(GTMOTransformations.alienated_number_operation(INF, Z), PHI)
        self.assertEqual(GTMOTransformations.alienated_number_operation(PHI, O), PHI)
        self.assertEqual(GTMOTransformations.alienated_number_operation(PSI, PSI), PHI)

        # Operations on definite states do not cause collapse
        self.assertEqual(GTMOTransformations.alienated_number_operation(O, O), O)
        self.assertEqual(GTMOTransformations.alienated_number_operation(Z, Z), Z)
        self.assertEqual(GTMOTransformations.alienated_number_operation(O, Z), Z)

class TestInterpretationEngine(unittest.TestCase):
    """Tests the engine for managing and analyzing multiple interpretations."""
    def setUp(self):
        """Prepares a fresh engine instance before each test."""
        self.logic = SANB5Logic()
        self.engine = InterpretationEngine(self.logic)

    def test_add_interpretation(self):
        """Checks if interpretations are added correctly."""
        self.assertEqual(len(self.engine.interpretations), 0)
        self.engine.add_interpretation("Test 1", O)
        self.assertEqual(len(self.engine.interpretations), 1)
        self.engine.add_interpretation("Test 2", PSI)
        self.assertEqual(len(self.engine.interpretations), 2)
        self.assertEqual(self.engine.interpretations[1], ("Test 2", PSI))

    def test_analyze_consistency(self):
        """Tests consistency analysis for various sets of interpretations."""
        # No interpretations -> Ø
        self.assertEqual(self.engine.analyze_consistency(), PHI)

        # One interpretation -> that interpretation
        self.engine.add_interpretation("a", INF)
        self.assertEqual(self.engine.analyze_consistency(), INF)

        # All the same -> that value
        self.engine.add_interpretation("b", INF)
        self.engine.add_interpretation("c", INF)
        self.assertEqual(self.engine.analyze_consistency(), INF)

        # Mix of O and Z -> contradiction -> Ø
        self.engine.interpretations = [("a", O), ("b", Z)]
        self.assertEqual(self.engine.analyze_consistency(), PHI)

        # Presence of INF -> chaos dominates
        self.engine.interpretations = [("a", O), ("b", Z), ("c", INF)]
        self.assertEqual(self.engine.analyze_consistency(), INF)

        # Other mixtures -> superposition
        self.engine.interpretations = [("a", O), ("b", PSI), ("c", Z)]
        self.assertEqual(self.engine.analyze_consistency(), PSI)

    def test_trajectory_productivity(self):
        """Tests the assessment of interpretation trajectory productivity."""
        # Insufficient data
        self.assertEqual(self.engine.trajectory_productivity(), "Insufficient data")
        self.engine.add_interpretation("start", PSI)
        self.assertEqual(self.engine.trajectory_productivity(), "Insufficient data")
        
        # Productive trajectory (towards definite states)
        self.engine.add_interpretation("step 1", O)
        self.engine.add_interpretation("step 2", Z)
        self.assertEqual(self.engine.trajectory_productivity(), "Productive trajectory")

        # Adding INF. Last 3 steps are [O, Z, INF].
        # Definite count (2) > chaos count (1) -> trajectory is still productive.
        self.engine.add_interpretation("step 3", INF)
        self.assertEqual(self.engine.trajectory_productivity(), "Productive trajectory") 
        
        # Unproductive trajectory
        self.engine.add_interpretation("step 4", INF)
        # Last 3 steps are [Z, INF, INF]. Chaos count (2) > definite count (1)
        self.assertEqual(self.engine.trajectory_productivity(), "Unproductive trajectory")

        # Neutral trajectory
        self.engine.interpretations = [("a", PSI), ("b", PHI), ("c", PSI)]
        self.assertEqual(self.engine.trajectory_productivity(), "Neutral trajectory")

def run_performance_tests():
    """Runs performance tests for key operations."""
    print("\n" + "=" * 50)
    print("SANB-5 LOGIC PERFORMANCE TESTS")
    print("=" * 50)
    
    # FIX: Define a globals dictionary for timeit to avoid import issues.
    test_globals = {
        'SANB5Logic': SANB5Logic,
        'GTMOTransformations': GTMOTransformations,
        'InterpretationEngine': InterpretationEngine,
        'SANB5Value': SANB5Value
    }
    
    # FIX: Remove import from __main__ from the setup code.
    setup_code = """
logic = SANB5Logic()
transform = GTMOTransformations()
engine = InterpretationEngine(logic)
values = list(SANB5Value)
for i in range(10):
    engine.add_interpretation(f"Interpretation {i}", values[i % 5])
"""
    iterations = 1_000_000

    perf_and = timeit.timeit(stmt="logic.and_(SANB5Value.O, SANB5Value.PSI)", setup=setup_code, number=iterations, globals=test_globals)
    print(f"1. Logical Operation (AND):")
    print(f"   - Total time for {iterations:,} operations: {perf_and:.4f} s")
    print(f"   - Average time per operation: {perf_and / iterations * 1e9:.2f} ns")

    perf_transform = timeit.timeit(stmt="transform.question_transformation(SANB5Value.O)", setup=setup_code, number=iterations, globals=test_globals)
    print(f"\n2. GTMØ Transformation (Question):")
    print(f"   - Total time for {iterations:,} operations: {perf_transform:.4f} s")
    print(f"   - Average time per operation: {perf_transform / iterations * 1e9:.2f} ns")

    iterations_engine = 100_000
    perf_engine = timeit.timeit(stmt="engine.analyze_consistency()", setup=setup_code, number=iterations_engine, globals=test_globals)
    print(f"\n3. Consistency Analysis (InterpretationEngine):")
    print(f"   - Total time for {iterations_engine:,} operations: {perf_engine:.4f} s")
    print(f"   - Average time per operation: {perf_engine / iterations_engine * 1e6:.2f} µs")

if __name__ == "__main__":
    print("Running unit tests...")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPhaseSpaceCoordinates))
    suite.addTest(unittest.makeSuite(TestSANB5Logic))
    suite.addTest(unittest.makeSuite(TestGTMOTransformations))
    suite.addTest(unittest.makeSuite(TestInterpretationEngine))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    
    # Run performance tests only if unit tests pass
    if result.wasSuccessful():
        run_performance_tests()
