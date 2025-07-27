"""
SANB-5 Logic Implementation for GTMØ Theory
===========================================

Implementation of 5-valued logic system (SANB-5) designed to work with
the Generalized Theory of Mathematical Indefiniteness (GTMØ).

Values:
- O (One/True): Definite truth, high determinacy, low entropy
- Z (Zero/False): Definite falsehood, high determinacy, low entropy  
- Ø (Singularity): Boundary of definition, where logic breaks down
- ∞ (Infinity/Chaos): Maximum entropy, all possibilities at once
- Ψ (Superposition): Coherent superposition of meanings

Author: SANB-5/GTMØ Integration
Version: 1.0
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, Callable
import numpy as np


class SANB5Value(Enum):
    """Five logical values in SANB-5 system."""
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
        """Convert SANB-5 value to Phase Space coordinates."""
        mapping = {
            SANB5Value.O: cls(0.95, 0.95, 0.05),    # High determinacy/stability, low entropy
            SANB5Value.Z: cls(0.95, 0.95, 0.05),    # High determinacy/stability, low entropy
            SANB5Value.PHI: cls(0.01, 0.01, 0.01),  # Singularity point
            SANB5Value.INF: cls(0.05, 0.05, 0.95),  # Low determinacy/stability, high entropy
            SANB5Value.PSI: cls(0.50, 0.30, 0.70),  # Medium values, potential for collapse
        }
        return mapping[value]


class SANB5Logic:
    """Implementation of SANB-5 logical operators."""
    
    def __init__(self):
        """Initialize SANB-5 logic system with truth tables."""
        self._init_truth_tables()
        
    def _init_truth_tables(self):
        """Initialize all truth tables for SANB-5 operators."""
        # Negation (¬)
        self.negation_table = {
            SANB5Value.O: SANB5Value.Z,
            SANB5Value.Z: SANB5Value.O,
            SANB5Value.PHI: SANB5Value.PHI,  # Negation of singularity is still singularity
            SANB5Value.INF: SANB5Value.INF,  # Negation of chaos is still chaos
            SANB5Value.PSI: SANB5Value.PSI,  # Negation of metaphor is another metaphor
        }
        
        # Conjunction (∧) - "Weakest Link"
        self.conjunction_table = self._create_conjunction_table()
        
        # Disjunction (∨) - "Strongest Link"
        self.disjunction_table = self._create_disjunction_table()
        
        # Implication (→)
        self.implication_table = self._create_implication_table()
        
        # Equivalence (↔)
        self.equivalence_table = self._create_equivalence_table()
        
        # XOR (⊕)
        self.xor_table = self._create_xor_table()
    
    def _create_conjunction_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        """Create conjunction truth table - weakest link principle."""
        table = {}
        values = list(SANB5Value)
        
        for v1 in values:
            for v2 in values:
                # Z (False) absorbs everything
                if v1 == SANB5Value.Z or v2 == SANB5Value.Z:
                    table[(v1, v2)] = SANB5Value.Z
                # Ø (Singularity) absorbs everything except Z
                elif v1 == SANB5Value.PHI or v2 == SANB5Value.PHI:
                    table[(v1, v2)] = SANB5Value.PHI
                # Both same value
                elif v1 == v2:
                    table[(v1, v2)] = v1
                # Mixed cases
                else:
                    if SANB5Value.INF in (v1, v2):
                        table[(v1, v2)] = SANB5Value.INF
                    else:
                        table[(v1, v2)] = SANB5Value.PSI
        
        return table
    
    def _create_disjunction_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        """Create disjunction truth table - strongest link principle."""
        table = {}
        values = list(SANB5Value)
        
        for v1 in values:
            for v2 in values:
                # O (True) dominates everything
                if v1 == SANB5Value.O or v2 == SANB5Value.O:
                    table[(v1, v2)] = SANB5Value.O
                # ∞ (Chaos) dominates everything except O
                elif v1 == SANB5Value.INF or v2 == SANB5Value.INF:
                    table[(v1, v2)] = SANB5Value.INF
                # Both same value
                elif v1 == v2:
                    table[(v1, v2)] = v1
                # Mixed cases
                else:
                    if SANB5Value.PSI in (v1, v2):
                        table[(v1, v2)] = SANB5Value.PSI
                    else:
                        table[(v1, v2)] = SANB5Value.PHI
        
        return table
    
    def _create_implication_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        """Create implication truth table - flow of certainty."""
        table = {}
        values = list(SANB5Value)
        
        for v1 in values:
            for v2 in values:
                # From Z (False) anything follows
                if v1 == SANB5Value.Z:
                    table[(v1, v2)] = SANB5Value.O
                # From O (True) the result depends on consequent
                elif v1 == SANB5Value.O:
                    table[(v1, v2)] = v2
                # From Ø, preserve singularity unless v2 is O
                elif v1 == SANB5Value.PHI:
                    table[(v1, v2)] = SANB5Value.O if v2 == SANB5Value.O else SANB5Value.PHI
                # From ∞, chaos propagates unless v2 is O
                elif v1 == SANB5Value.INF:
                    table[(v1, v2)] = SANB5Value.O if v2 == SANB5Value.O else SANB5Value.INF
                # From Ψ, superposition propagates unless v2 is O
                elif v1 == SANB5Value.PSI:
                    table[(v1, v2)] = SANB5Value.O if v2 == SANB5Value.O else SANB5Value.PSI
        
        return table
    
    def _create_equivalence_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        """Create equivalence truth table - ontological identity test."""
        table = {}
        values = list(SANB5Value)
        
        for v1 in values:
            for v2 in values:
                if v1 == v2:
                    # Same definite values are equivalent
                    if v1 in (SANB5Value.O, SANB5Value.Z, SANB5Value.PSI):
                        table[(v1, v2)] = SANB5Value.O
                    # Singularity and chaos remain themselves
                    else:
                        table[(v1, v2)] = v1
                else:
                    # Different values - result depends on types
                    if {v1, v2} == {SANB5Value.O, SANB5Value.Z}:
                        table[(v1, v2)] = SANB5Value.Z
                    elif SANB5Value.PHI in (v1, v2):
                        table[(v1, v2)] = SANB5Value.PHI
                    elif SANB5Value.INF in (v1, v2):
                        table[(v1, v2)] = SANB5Value.INF
                    else:
                        table[(v1, v2)] = SANB5Value.PSI
        
        return table
    
    def _create_xor_table(self) -> Dict[Tuple[SANB5Value, SANB5Value], SANB5Value]:
        """Create XOR truth table - rigorous difference test."""
        table = {}
        values = list(SANB5Value)
        
        for v1 in values:
            for v2 in values:
                if v1 == v2:
                    # Same values are not exclusively different
                    if v1 in (SANB5Value.O, SANB5Value.Z, SANB5Value.PSI):
                        table[(v1, v2)] = SANB5Value.Z
                    else:
                        table[(v1, v2)] = v1
                else:
                    # Different values
                    if {v1, v2} == {SANB5Value.O, SANB5Value.Z}:
                        table[(v1, v2)] = SANB5Value.O  # Classic XOR
                    elif SANB5Value.PHI in (v1, v2):
                        table[(v1, v2)] = SANB5Value.PHI
                    elif SANB5Value.INF in (v1, v2):
                        table[(v1, v2)] = SANB5Value.INF
                    elif SANB5Value.O in (v1, v2):
                        table[(v1, v2)] = SANB5Value.O
                    else:
                        table[(v1, v2)] = SANB5Value.PSI
        
        return table
    
    # Operator methods
    def neg(self, a: SANB5Value) -> SANB5Value:
        """Negation operator (¬)."""
        return self.negation_table[a]
    
    def and_(self, a: SANB5Value, b: SANB5Value) -> SANB5Value:
        """Conjunction operator (∧)."""
        return self.conjunction_table[(a, b)]
    
    def or_(self, a: SANB5Value, b: SANB5Value) -> SANB5Value:
        """Disjunction operator (∨)."""
        return self.disjunction_table[(a, b)]
    
    def implies(self, a: SANB5Value, b: SANB5Value) -> SANB5Value:
        """Implication operator (→)."""
        return self.implication_table[(a, b)]
    
    def equiv(self, a: SANB5Value, b: SANB5Value) -> SANB5Value:
        """Equivalence operator (↔)."""
        return self.equivalence_table[(a, b)]
    
    def xor(self, a: SANB5Value, b: SANB5Value) -> SANB5Value:
        """XOR operator (⊕)."""
        return self.xor_table[(a, b)]


class GTMOTransformations:
    """Transformations between SANB-5 and GTMØ concepts."""
    
    @staticmethod
    def question_transformation(value: SANB5Value) -> SANB5Value:
        """
        Models GTMØ's question transformation (0→0?, 1→1?).
        Transforms definite states into superpositions.
        """
        if value in (SANB5Value.O, SANB5Value.Z):
            return SANB5Value.PSI  # Definite becomes superposition
        else:
            return value  # Already indefinite states remain unchanged
    
    @staticmethod
    def observation_collapse(value: SANB5Value, observer_bias: float = 0.5) -> SANB5Value:
        """
        Models collapse of superposition through observation.
        observer_bias: 0.0 = tends toward Z, 1.0 = tends toward O
        """
        if value == SANB5Value.PSI:
            if observer_bias < 0.2:
                return SANB5Value.Z
            elif observer_bias > 0.8:
                return SANB5Value.O
            elif observer_bias < 0.3:
                return SANB5Value.PHI
            elif observer_bias > 0.7:
                return SANB5Value.INF
            else:
                return SANB5Value.PSI  # Remains in superposition
        return value
    
    @staticmethod
    def alienated_number_operation(v1: SANB5Value, v2: SANB5Value) -> SANB5Value:
        """
        Models GTMØ's AlienatedNumber arithmetic collapse.
        Pure mathematical operations on indefinite entities yield Ø.
        """
        # If either operand has indefinite aspects, result collapses to Ø
        if v1 in (SANB5Value.PHI, SANB5Value.INF, SANB5Value.PSI) or \
           v2 in (SANB5Value.PHI, SANB5Value.INF, SANB5Value.PSI):
            return SANB5Value.PHI
        # Otherwise standard arithmetic on definite values
        elif v1 == SANB5Value.O and v2 == SANB5Value.O:
            return SANB5Value.O  # 1 + 1 = 2 (still true)
        else:
            return SANB5Value.Z


class InterpretationEngine:
    """Engine for handling multiple interpretations in SANB-5."""
    
    def __init__(self, logic: SANB5Logic):
        self.logic = logic
        self.interpretations: List[Tuple[str, SANB5Value]] = []
    
    def add_interpretation(self, description: str, value: SANB5Value):
        """Add a new interpretation to the collection."""
        self.interpretations.append((description, value))
    
    def analyze_consistency(self) -> SANB5Value:
        """
        Analyze consistency of all interpretations.
        Returns the dominant logical value across interpretations.
        """
        if not self.interpretations:
            return SANB5Value.PHI  # No interpretations = singularity
        
        values = [v for _, v in self.interpretations]
        
        # All same = that value
        if len(set(values)) == 1:
            return values[0]
        
        # Mix of O and Z = contradiction = Ø
        if set(values) == {SANB5Value.O, SANB5Value.Z}:
            return SANB5Value.PHI
        
        # Any ∞ = chaos dominates
        if SANB5Value.INF in values:
            return SANB5Value.INF
        
        # Otherwise superposition
        return SANB5Value.PSI
    
    def trajectory_productivity(self) -> str:
        """
        Assess if interpretation trajectory is productive (per GTMØ).
        Productive: moves toward definiteness (O/Z)
        Unproductive: moves toward chaos (∞)
        """
        if len(self.interpretations) < 2:
            return "Insufficient data"
        
        # Check trend in last few interpretations
        recent = self.interpretations[-3:]
        recent_values = [v for _, v in recent]
        
        definite_count = sum(1 for v in recent_values if v in (SANB5Value.O, SANB5Value.Z))
        chaos_count = sum(1 for v in recent_values if v == SANB5Value.INF)
        
        if definite_count > chaos_count:
            return "Productive trajectory"
        elif chaos_count > definite_count:
            return "Unproductive trajectory"
        else:
            return "Neutral trajectory"


# Example usage and demonstrations
def demonstrate_sanb5():
    """Demonstrate SANB-5 logic with GTMØ integration."""
    logic = SANB5Logic()
    transform = GTMOTransformations()
    
    print("SANB-5 Logic Demonstration")
    print("=" * 50)
    
    # Basic operations
    print("\n1. Basic Operations:")
    print(f"¬O = {logic.neg(SANB5Value.O)}")
    print(f"¬Ψ = {logic.neg(SANB5Value.PSI)} (negation of metaphor is still metaphor)")
    print(f"O ∧ Ψ = {logic.and_(SANB5Value.O, SANB5Value.PSI)}")
    print(f"Z ∨ ∞ = {logic.or_(SANB5Value.Z, SANB5Value.INF)}")
    
    # GTMØ transformations
    print("\n2. GTMØ Transformations:")
    print(f"Question transformation O → {transform.question_transformation(SANB5Value.O)}")
    print(f"Question transformation Z → {transform.question_transformation(SANB5Value.Z)}")
    
    # AlienatedNumber collapse
    print("\n3. AlienatedNumber Operations:")
    print(f"O + Ψ = {transform.alienated_number_operation(SANB5Value.O, SANB5Value.PSI)} (collapse to Ø)")
    
    # Interpretation example
    print("\n4. Multiple Interpretations (Paper Experiment):")
    engine = InterpretationEngine(logic)
    
    # From GTMØ paper experiment
    engine.add_interpretation("0+1 = 1 (arithmetic)", SANB5Value.O)
    engine.add_interpretation("0+1 = 01 (concatenation)", SANB5Value.PSI)
    engine.add_interpretation("0+1 = 10 (reverse concat)", SANB5Value.PSI)
    engine.add_interpretation("0+1 = 2 (counting papers)", SANB5Value.O)
    
    print(f"Consistency analysis: {engine.analyze_consistency()}")
    print(f"Trajectory assessment: {engine.trajectory_productivity()}")
    
    # Phase space mapping
    print("\n5. Phase Space Mapping:")
    for value in SANB5Value:
        coords = PhaseSpaceCoordinates.from_sanb5(value)
        print(f"{value} → (det={coords.determinacy:.2f}, "
              f"stab={coords.stability:.2f}, ent={coords.entropy:.2f})")


if __name__ == "__main__":
    demonstrate_sanb5()
