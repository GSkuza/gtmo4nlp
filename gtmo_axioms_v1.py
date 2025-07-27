# gtmo/gtmo_axioms_v1.py

"""gtmo_axioms_v1.py
----------------------------------

- Dynamic context-aware operators instead of simple heuristics
- Executable axioms that can transform system state
- Topological classification with attractors instead of percentage thresholds
- Adaptive learning capabilities for neurons
- Enhanced system integration with learning and defense mechanisms
- Preserved UniverseMode functionality from original
"""

from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import random
import logging

# Import enhanced GTMØ core components from v2
try:
    from gtmo_core_v2 import (
        O, AlienatedNumber, Singularity, STRICT_MODE, SingularityError,
        ExecutableAxiom, TopologicalClassifier, AdaptiveGTMONeuron,
        KnowledgeEntity, KnowledgeType, EpistemicParticle, GTMOSystemV2,
        AX0_SystemicUncertainty, AX1_OntologicalDifference, AX6_MinimalEntropy
    )
    V2_AVAILABLE = True
except ImportError:
    # Fallback to basic core if v2 not available
    from core import O, AlienatedNumber, Singularity, STRICT_MODE, SingularityError
    V2_AVAILABLE = False
    print("Warning: gtmo-core-v2.py not available, using basic functionality")

# Set up logging for GTMØ operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


###############################################################################
# Enhanced GTMØ Formal Axioms and Definitions
###############################################################################

class GTMOAxiom:
    """Container for GTMØ formal axioms with validation capabilities."""

    # Formal axioms (AX0-AX10) - Extended with v2 context
    AX0 = "Systemic Uncertainty: There is no proof that the GTMØ system is fully definable, and its foundational state (e.g., stillness vs. flux) must be axiomatically assumed."
    AX1 = "Ø is a fundamentally different mathematical category: Ø ∉ {0, 1, ∞} ∧ ¬∃f, D: f(D) = Ø, D ⊆ {0,1,∞}"
    AX2 = "Translogical isolation: ¬∃f: D → Ø, D ⊆ DefinableSystems"
    AX3 = "Epistemic singularity: ¬∃S: Know(Ø) ∈ S, S ∈ CognitiveSystems"
    AX4 = "Non-representability: Ø ∉ Repr(S), ∀S ⊇ {0,1,∞}"
    AX5 = "Topological boundary: Ø ∈ ∂(CognitiveSpace)"
    AX6 = "Heuristic extremum: E_GTMØ(Ø) = min E_GTMØ(x), x ∈ KnowledgeDomain"
    AX7 = "Meta-closure: Ø ∈ MetaClosure(GTMØ) ∧ Ø triggers system self-evaluation"
    AX8 = "Ø is not a topological limit point: ¬∃Seq(xₙ) ⊆ Domain(GTMØ): lim(xₙ) = Ø"
    AX9 = "Operator irreducibility (strict): ¬∃Op ∈ StandardOperators: Op(Ø) = x, x ∈ Domain(GTMØ)"
    AX10 = "Meta-operator definition: Ψ_GTMØ, E_GTMØ are meta-operators acting on Ø"
    
    # NEW: v2 axioms for adaptive learning
    AX11 = "Adaptive learning: System neurons can modify their response patterns based on adversarial experience"
    AX12 = "Topological phase classification: Knowledge types correspond to basins of attraction in phase space"
    
    ALL_AXIOMS = [AX0, AX1, AX2, AX3, AX4, AX5, AX6, AX7, AX8, AX9, AX10, AX11, AX12]
    
    @classmethod
    def validate_axiom_compliance(cls, operation_result: Any, axiom_id: str) -> bool:
        """Enhanced validation with v2 capabilities."""
        if axiom_id == "AX0":
            return True  # Meta-axiom about system nature
        elif axiom_id == "AX1":
            return operation_result not in {0, 1, float('inf'), -float('inf')}
        elif axiom_id == "AX6":
            return hasattr(operation_result, 'entropy') and operation_result.entropy <= 0.001
        elif axiom_id == "AX9":
            return isinstance(operation_result, (SingularityError, ValueError))
        elif axiom_id == "AX11":
            # Check if system has learning neurons
            return hasattr(operation_result, 'neurons') and any(
                hasattr(n, 'long_term_memory') for n in getattr(operation_result, 'neurons', [])
            )
        elif axiom_id == "AX12":
            # Check if system uses topological classification
            return hasattr(operation_result, 'classifier') and hasattr(
                getattr(operation_result, 'classifier', None), 'attractors'
            )
        else:
            return True


class GTMODefinition:
    """Enhanced GTMØ formal definitions with v2 concepts."""
    
    DEF1 = "Knowledge particle Ψᴷ – a fragment in the attractor basin of high determinacy-stability"
    DEF2 = "Knowledge shadow Ψʰ – a fragment in the attractor basin of low determinacy-stability" 
    DEF3 = "Cognitive entropy E_GTMØ(x) = context-aware semantic partitioning entropy"
    DEF4 = "Novel emergent type Ψᴺ – fragments exhibiting unbounded epistemic expansion"
    DEF5 = "Liminal type Ψᴧ – fragments in phase space regions between attractor basins"
    DEF6 = "Adaptive neuron – a processing unit capable of learning defense strategies from experience"
    DEF7 = "Topological attractor – regions in phase space that attract similar knowledge types"
    
    ALL_DEFINITIONS = [DEF1, DEF2, DEF3, DEF4, DEF5, DEF6, DEF7]


###############################################################################
# Enhanced Operators with v2 Integration
###############################################################################

class OperatorType(Enum):
    """Types of operators in GTMØ framework."""
    STANDARD = 1
    META = 2
    HYBRID = 3
    ADAPTIVE = 4  # NEW: for learning operators


class OperationResult:
    """Enhanced container for GTMØ operation results with v2 metadata."""
    
    def __init__(
        self,
        value: Any,
        operator_type: OperatorType,
        axiom_compliance: Dict[str, bool] = None,
        metadata: Dict[str, Any] = None,
        learning_data: Dict[str, Any] = None  # NEW: for adaptive learning
    ):
        self.value = value
        self.operator_type = operator_type
        self.axiom_compliance = axiom_compliance or {}
        self.metadata = metadata or {}
        self.learning_data = learning_data or {}  # Store experience for learning
        
    def __repr__(self) -> str:
        return f"OperationResult(value={self.value}, type={self.operator_type.name})"


class EnhancedPsiOperator:
    """Enhanced Ψ_GTMØ operator using v2 dynamic context-aware calculations."""
    
    def __init__(self, classifier: 'TopologicalClassifier' = None):
        self.classifier = classifier or (TopologicalClassifier() if V2_AVAILABLE else None)
        self.operation_count = 0
        self.adaptation_history = []
    
    def __call__(self, fragment: Any, context: Dict[str, Any] = None) -> OperationResult:
        self.operation_count += 1
        context = context or {}
        
        if fragment is O:
            return self._process_singularity(context)
        elif isinstance(fragment, AlienatedNumber):
            return self._process_alienated_number_v2(fragment, context)
        else:
            return self._process_general_fragment_v2(fragment, context)
    
    def _process_singularity(self, context: Dict[str, Any]) -> OperationResult:
        """Ø processing remains the same - it's the fixed point."""
        return OperationResult(
            value={
                'score': 1.0, 
                'type': 'Ø (ontological_singularity)', 
                'classification': 'Ø', 
                'meta_operator_applied': True
            },
            operator_type=OperatorType.META,
            axiom_compliance={'AX6': True, 'AX10': True},
            metadata={
                'processed_by': 'Enhanced_Ψ_GTMØ_meta',
                'singularity_detected': True,
                'operation_id': self.operation_count
            }
        )
    
    def _process_alienated_number_v2(self, alienated_num: AlienatedNumber, 
                                    context: Dict[str, Any]) -> OperationResult:
        """Enhanced processing using v2 dynamic context-aware calculations."""
        # Use v2's dynamic calculation instead of fixed 0.999
        psi_score = alienated_num.psi_gtm_score()
        
        # Extract context information for metadata
        semantic_distance = getattr(alienated_num, '_semantic_cache', {}).get('semantic_distance', 0)
        
        return OperationResult(
            value={
                'score': psi_score,
                'type': f'ℓ∅ ({alienated_num.identifier})',
                'classification': 'ℓ∅',
                'meta_operator_applied': True,
                'context_factors': {
                    'semantic_distance': semantic_distance,
                    'has_context': bool(alienated_num.context)
                }
            },
            operator_type=OperatorType.META,
            metadata={
                'alienated_identifier': alienated_num.identifier,
                'operation_id': self.operation_count,
                'context_aware': True
            }
        )
    
    def _process_general_fragment_v2(self, fragment: Any, context: Dict[str, Any]) -> OperationResult:
        """Enhanced general processing with topological classification."""
        if V2_AVAILABLE and self.classifier:
            # Use topological classification instead of thresholds
            if isinstance(fragment, KnowledgeEntity):
                classification_type = self.classifier.classify(fragment)
                phase_point = fragment.to_phase_point()
                score = np.mean(phase_point[:2])  # Average of determinacy and stability
            else:
                # Create temporary entity for classification
                temp_entity = self._create_temp_entity(fragment)
                classification_type = self.classifier.classify(temp_entity)
                phase_point = temp_entity.to_phase_point()
                score = np.mean(phase_point[:2])
            
            return OperationResult(
                value={
                    'score': score,
                    'type': f'{classification_type.value}',
                    'classification': classification_type.value,
                    'phase_point': phase_point,
                    'topological_classification': True
                },
                operator_type=OperatorType.ADAPTIVE,
                metadata={
                    'fragment_type': type(fragment).__name__,
                    'operation_id': self.operation_count,
                    'classifier_used': 'TopologicalClassifier'
                }
            )
        else:
            # Fallback to simple heuristic if v2 not available
            score = self._calculate_epistemic_purity_fallback(fragment)
            return OperationResult(
                value={
                    'score': score,
                    'type': 'heuristic_classification',
                    'classification': 'Ψᴧ' if 0.3 < score < 0.7 else ('Ψᴷ' if score > 0.7 else 'Ψʰ')
                },
                operator_type=OperatorType.STANDARD,
                metadata={
                    'fragment_type': type(fragment).__name__,
                    'operation_id': self.operation_count,
                    'fallback_mode': True
                }
            )
    
    def _create_temp_entity(self, fragment: Any) -> 'KnowledgeEntity':
        """Create temporary KnowledgeEntity for classification."""
        if not V2_AVAILABLE:
            return None
        
        # Estimate properties from content
        content_str = str(fragment).lower()
        
        determinacy = 0.5
        stability = 0.5
        entropy = 0.5
        
        # Enhanced heuristics
        if any(word in content_str for word in ['certain', 'always', 'never', 'theorem', 'proof']):
            determinacy += 0.3
            stability += 0.2
            entropy -= 0.3
        
        if any(word in content_str for word in ['maybe', 'possibly', 'uncertain', 'might']):
            determinacy -= 0.3
            entropy += 0.3
        
        if any(word in content_str for word in ['paradox', 'contradiction', 'impossible']):
            stability -= 0.4
            entropy += 0.4
        
        # Normalize
        determinacy = max(0, min(1, determinacy))
        stability = max(0, min(1, stability))
        entropy = max(0, min(1, entropy))
        
        return KnowledgeEntity(
            content=fragment,
            determinacy=determinacy,
            stability=stability,
            entropy=entropy
        )
    
    def _calculate_epistemic_purity_fallback(self, fragment: Any) -> float:
        """Fallback heuristic calculation when v2 not available."""
        fragment_str = str(fragment).lower()
        score = 0.5
        
        if any(keyword in fragment_str for keyword in ['theorem', 'proof', 'axiom', 'definition']):
            score += 0.2
        if any(keyword in fragment_str for keyword in ['is', 'equals', 'always', 'never']):
            score += 0.1
        if any(keyword in fragment_str for keyword in ['maybe', 'perhaps', 'might', 'could']):
            score -= 0.2
        if any(keyword in fragment_str for keyword in ['paradox', 'contradiction', 'impossible']):
            score -= 0.3
        
        return max(0.0, min(1.0, score))


class EnhancedEntropyOperator:
    """Enhanced E_GTMØ operator using v2 context-aware entropy calculations."""
    
    def __init__(self):
        self.operation_count = 0
    
    def __call__(self, fragment: Any, context: Dict[str, Any] = None) -> OperationResult:
        self.operation_count += 1
        context = context or {}
        
        if fragment is O:
            return self._process_singularity_entropy(context)
        elif isinstance(fragment, AlienatedNumber):
            return self._process_alienated_entropy_v2(fragment, context)
        else:
            return self._process_general_entropy_v2(fragment, context)
    
    def _process_singularity_entropy(self, context: Dict[str, Any]) -> OperationResult:
        """Ø has minimal entropy (AX6)."""
        return OperationResult(
            value={
                'total_entropy': 0.0,
                'Ψᴷ_entropy': 0.0,
                'Ψʰ_entropy': 0.0,
                'partitions': [1.0],
                'explanation': 'Ø has minimal cognitive entropy (AX6)'
            },
            operator_type=OperatorType.META,
            axiom_compliance={'AX6': True},
            metadata={
                'singularity_processed': True,
                'operation_id': self.operation_count
            }
        )
    
    def _process_alienated_entropy_v2(self, alienated_num: AlienatedNumber, 
                                     context: Dict[str, Any]) -> OperationResult:
        """Enhanced processing using v2 dynamic entropy calculation."""
        # Use v2's context-aware entropy calculation
        entropy_value = alienated_num.e_gtm_entropy()
        
        # Extract additional context information
        context_factors = {
            'temporal_distance': alienated_num.context.get('temporal_distance', 0),
            'volatility': alienated_num.context.get('volatility', 0),
            'predictability': alienated_num.context.get('predictability', 1)
        }
        
        return OperationResult(
            value={
                'total_entropy': entropy_value,
                'Ψᴷ_entropy': entropy_value * 0.1,
                'Ψʰ_entropy': entropy_value * 0.9,
                'partitions': [0.1, 0.9],
                'explanation': f'Context-aware entropy for {alienated_num.identifier}',
                'context_factors': context_factors
            },
            operator_type=OperatorType.META,
            metadata={
                'alienated_identifier': alienated_num.identifier,
                'operation_id': self.operation_count,
                'context_aware': True
            }
        )
    
    def _process_general_entropy_v2(self, fragment: Any, context: Dict[str, Any]) -> OperationResult:
        """Enhanced general entropy processing."""
        if V2_AVAILABLE and isinstance(fragment, KnowledgeEntity):
            # Use phase space entropy
            phase_point = fragment.to_phase_point()
            entropy = phase_point[2]  # Entropy coordinate
            
            # Calculate partitions based on phase space position
            partitions = self._calculate_phase_space_partitions(phase_point)
        else:
            # Fallback to semantic partitioning
            partitions = self._calculate_semantic_partitions_fallback(fragment)
            entropy = -sum(p * math.log2(p) for p in partitions if p > 0)
        
        total_entropy = entropy
        psi_k_entropy = -partitions[0] * math.log2(partitions[0]) if partitions[0] > 0 else 0
        psi_h_entropy = -partitions[-1] * math.log2(partitions[-1]) if partitions[-1] > 0 else 0
        
        return OperationResult(
            value={
                'total_entropy': total_entropy,
                'Ψᴷ_entropy': psi_k_entropy,
                'Ψʰ_entropy': psi_h_entropy,
                'partitions': partitions,
                'explanation': f'Enhanced entropy calculation: {total_entropy:.3f}'
            },
            operator_type=OperatorType.ADAPTIVE if V2_AVAILABLE else OperatorType.STANDARD,
            metadata={
                'partition_count': len(partitions),
                'operation_id': self.operation_count,
                'method': 'phase_space' if V2_AVAILABLE else 'semantic'
            }
        )
    
    def _calculate_phase_space_partitions(self, phase_point: Tuple[float, float, float]) -> List[float]:
        """Calculate partitions based on phase space coordinates."""
        determinacy, stability, entropy = phase_point
        
        # Partition based on phase space regions
        knowledge_weight = determinacy * stability
        shadow_weight = (1 - determinacy) * (1 - stability)
        liminal_weight = 1 - knowledge_weight - shadow_weight
        
        total = knowledge_weight + shadow_weight + liminal_weight
        if total > 0:
            partitions = [knowledge_weight/total, liminal_weight/total, shadow_weight/total]
        else:
            partitions = [0.33, 0.33, 0.34]
        
        # Ensure minimum values
        partitions = [max(p, 0.001) for p in partitions]
        total = sum(partitions)
        partitions = [p/total for p in partitions]
        
        return partitions
    
    def _calculate_semantic_partitions_fallback(self, fragment: Any) -> List[float]:
        """Fallback semantic partitioning when v2 not available."""
        fragment_str = str(fragment).lower()
        certain_weight, uncertain_weight, unknown_weight = 0.4, 0.4, 0.2
        
        certainty_count = sum(1 for ind in ['is', 'equals', 'always', 'never', 'theorem'] if ind in fragment_str)
        uncertainty_count = sum(1 for ind in ['maybe', 'perhaps', 'might', 'could'] if ind in fragment_str)
        paradox_count = sum(1 for ind in ['paradox', 'contradiction', 'impossible'] if ind in fragment_str)
        
        if certainty_count > 0:
            certain_weight += 0.2 * certainty_count
            uncertain_weight -= 0.1 * certainty_count
        if uncertainty_count > 0:
            uncertain_weight += 0.2 * uncertainty_count
            certain_weight -= 0.1 * uncertainty_count
        if paradox_count > 0:
            unknown_weight += 0.3 * paradox_count
            certain_weight -= 0.15 * paradox_count
            uncertain_weight -= 0.15 * paradox_count
        
        total = certain_weight + uncertain_weight + unknown_weight
        partitions = [certain_weight/total, uncertain_weight/total, unknown_weight/total]
        partitions = [max(p, 0.001) for p in partitions]
        total = sum(partitions)
        partitions = [p/total for p in partitions]
        
        return partitions


###############################################################################
# Enhanced Meta-Feedback Loop with v2 Integration
###############################################################################

class EnhancedMetaFeedbackLoop:
    """Enhanced meta-feedback loop integrating v2 adaptive learning capabilities."""
    
    def __init__(self, psi_operator: EnhancedPsiOperator, entropy_operator: EnhancedEntropyOperator):
        self.psi_operator = psi_operator
        self.entropy_operator = entropy_operator
        self.emergence_detector = EmergenceDetector()
        self.learning_history = []
        self.adaptation_weights = np.ones(5) * 0.2  # Adaptive weights for different factors
    
    def run(self, fragments: List[Any], initial_scores: List[float], 
            iterations: int = 5, learning_enabled: bool = True) -> Dict[str, Any]:
        """Enhanced feedback loop with learning capabilities."""
        history = []
        current_scores = list(initial_scores)
        new_types_detected = set()
        
        logger.info(f"Starting enhanced meta-feedback loop with {len(fragments)} fragments, {iterations} iterations")
        
        for iteration in range(iterations):
            iteration_data = self._process_iteration_v2(
                fragments, current_scores, iteration, new_types_detected, learning_enabled
            )
            history.append(iteration_data)
            
            # Update scores for next iteration
            new_scores = [item['score'] for item in iteration_data['fragment_results'] if item['score'] is not None]
            if new_scores:
                current_scores.extend(new_scores)
                current_scores = current_scores[-max(len(initial_scores), 100):]
            
            # Adaptive learning: adjust weights based on results
            if learning_enabled and iteration > 0:
                self._adapt_processing_weights(iteration_data, history[-2] if len(history) > 1 else None)
        
        final_state = self._analyze_final_state_v2(history, new_types_detected)
        
        # Store learning experience
        if learning_enabled:
            self.learning_history.append({
                'fragments_count': len(fragments),
                'iterations': iterations,
                'final_state': final_state,
                'adaptation_weights': self.adaptation_weights.copy()
            })
        
        return {
            'history': history,
            'final_state': final_state,
            'new_types_detected': list(new_types_detected),
            'learning_enabled': learning_enabled,
            'adaptation_weights': self.adaptation_weights.tolist()
        }
    
    def _process_iteration_v2(self, fragments: List[Any], current_scores: List[float],
                             iteration: int, new_types_detected: Set[str], 
                             learning_enabled: bool) -> Dict[str, Any]:
        """Enhanced iteration processing with v2 capabilities."""
        fragment_results = []
        iteration_scores = []
        iteration_types = []
        iteration_entropies = []
        
        context = {
            'all_scores': current_scores,
            'iteration': iteration,
            'timestamp': iteration * 0.1,
            'learning_enabled': learning_enabled,
            'adaptation_weights': self.adaptation_weights
        }
        
        for frag_idx, fragment in enumerate(fragments):
            # Enhanced processing with learning context
            psi_result = self.psi_operator(fragment, context)
            entropy_result = self.entropy_operator(fragment, context)
            
            score = psi_result.value.get('score')
            classification = psi_result.value.get('classification', 'unknown')
            total_entropy = entropy_result.value.get('total_entropy', 0.0)
            
            if score is not None:
                iteration_scores.append(score)
            iteration_types.append(classification)
            iteration_entropies.append(total_entropy)
            
            # Enhanced emergence detection
            emergence_result = self.emergence_detector.detect_emergence(fragment, psi_result, entropy_result)
            if emergence_result['is_emergent']:
                new_types_detected.add(emergence_result['emergent_type'])
            
            fragment_results.append({
                'fragment_index': frag_idx,
                'fragment': str(fragment)[:100],
                'score': score,
                'classification': classification,
                'entropy': total_entropy,
                'emergence': emergence_result,
                'v2_features': {
                    'topological_classification': psi_result.value.get('topological_classification', False),
                    'context_aware': psi_result.metadata.get('context_aware', False),
                    'phase_point': psi_result.value.get('phase_point')
                }
            })
        
        # Calculate enhanced metrics
        classification_counts = {cls: iteration_types.count(cls) for cls in set(iteration_types)}
        total_classifications = len(iteration_types)
        classification_ratios = {cls: count / total_classifications for cls, count in classification_counts.items()}
        
        return {
            'iteration': iteration,
            'fragment_results': fragment_results,
            'scores': iteration_scores,
            'types': iteration_types,
            'entropies': iteration_entropies,
            'classification_ratios': classification_ratios,
            'average_entropy': np.mean(iteration_entropies) if iteration_entropies else 0.0,
            'average_score': np.mean(iteration_scores) if iteration_scores else 0.0,
            'v2_metrics': {
                'topological_classifications': sum(1 for r in fragment_results 
                                                 if r['v2_features']['topological_classification']),
                'context_aware_operations': sum(1 for r in fragment_results 
                                              if r['v2_features']['context_aware']),
                'phase_space_coverage': self._calculate_phase_space_coverage(fragment_results)
            }
        }
    
    def _calculate_phase_space_coverage(self, fragment_results: List[Dict]) -> float:
        """Calculate how much of the phase space is covered by current fragments."""
        phase_points = [r['v2_features']['phase_point'] for r in fragment_results 
                       if r['v2_features']['phase_point'] is not None]
        
        if not phase_points:
            return 0.0
        
        # Simple coverage metric: volume of bounding box in phase space
        if len(phase_points) == 1:
            return 0.1  # Single point
        
        points_array = np.array(phase_points)
        ranges = np.max(points_array, axis=0) - np.min(points_array, axis=0)
        volume = np.prod(ranges)
        
        return min(1.0, volume)  # Normalize to [0, 1]
    
    def _adapt_processing_weights(self, current_iteration: Dict[str, Any], 
                                 previous_iteration: Optional[Dict[str, Any]]):
        """Adapt processing weights based on performance."""
        if not previous_iteration:
            return
        
        # Calculate performance metrics
        current_avg_score = current_iteration['average_score']
        previous_avg_score = previous_iteration['average_score']
        
        current_entropy = current_iteration['average_entropy']
        previous_entropy = previous_iteration['average_entropy']
        
        # Adapt weights based on trends
        score_improvement = current_avg_score - previous_avg_score
        entropy_change = current_entropy - previous_entropy
        
        # Simple adaptation rule: increase weights for improving factors
        if score_improvement > 0.01:
            self.adaptation_weights[0] = min(1.0, self.adaptation_weights[0] + 0.05)  # Score factor
        if abs(entropy_change) < 0.01:  # Stability in entropy is good
            self.adaptation_weights[1] = min(1.0, self.adaptation_weights[1] + 0.05)  # Stability factor
        
        # Normalize weights
        self.adaptation_weights = self.adaptation_weights / np.sum(self.adaptation_weights)
    
    def _analyze_final_state_v2(self, history: List[Dict[str, Any]], 
                               new_types_detected: Set[str]) -> Dict[str, Any]:
        """Enhanced final state analysis with v2 metrics."""
        if not history:
            return {'status': 'no_iterations_completed'}
        
        final_iteration = history[-1]
        
        # Standard metrics
        score_trend = [item['average_score'] for item in history]
        entropy_trend = [item['average_entropy'] for item in history]
        
        convergence_threshold = 0.01
        score_convergence = (len(score_trend) >= 3 and 
                           abs(score_trend[-1] - score_trend[-2]) < convergence_threshold)
        entropy_convergence = (len(entropy_trend) >= 3 and 
                             abs(entropy_trend[-1] - entropy_trend[-2]) < convergence_threshold)
        
        # Enhanced v2 metrics
        v2_metrics = {}
        if 'v2_metrics' in final_iteration:
            v2_metrics = {
                'final_topological_ratio': (final_iteration['v2_metrics']['topological_classifications'] / 
                                           len(final_iteration['fragment_results'])),
                'context_awareness_ratio': (final_iteration['v2_metrics']['context_aware_operations'] / 
                                          len(final_iteration['fragment_results'])),
                'phase_space_coverage': final_iteration['v2_metrics']['phase_space_coverage'],
                'coverage_trend': [h['v2_metrics']['phase_space_coverage'] for h in history 
                                 if 'v2_metrics' in h]
            }
        
        return {
            'final_classification_ratios': final_iteration['classification_ratios'],
            'score_convergence': score_convergence,
            'entropy_convergence': entropy_convergence,
            'system_stability': score_convergence and entropy_convergence,
            'total_emergent_types': len(new_types_detected),
            'score_trend': score_trend,
            'entropy_trend': entropy_trend,
            'iterations_completed': len(history),
            'v2_enhanced_metrics': v2_metrics,
            'adaptation_effectiveness': self._calculate_adaptation_effectiveness(history)
        }
    
    def _calculate_adaptation_effectiveness(self, history: List[Dict[str, Any]]) -> float:
        """Calculate how effective the adaptation was during the process."""
        if len(history) < 3:
            return 0.5  # Not enough data
        
        # Look at improvement over time
        early_scores = np.mean([h['average_score'] for h in history[:len(history)//2]])
        late_scores = np.mean([h['average_score'] for h in history[len(history)//2:]])
        
        improvement = late_scores - early_scores
        return max(0.0, min(1.0, 0.5 + improvement))  # Map to [0, 1]


###############################################################################
# Enhanced System with Universe Modes and v2 Integration
###############################################################################

class UniverseMode(Enum):
    """Universe modes from original gtmo_axioms.py preserved."""
    INDEFINITE_STILLNESS = auto()
    ETERNAL_FLUX = auto()


class EnhancedGTMOSystem:
    """Enhanced GTMØ system integrating v2 capabilities with original UniverseMode."""
    
    def __init__(self, mode: UniverseMode, initial_fragments: Optional[List[Any]] = None,
                 enable_v2_features: bool = True):
        """
        Initialize enhanced GTMØ system.
        
        Args:
            mode: Universe mode (Stillness vs Flux)
            initial_fragments: Initial knowledge fragments
            enable_v2_features: Whether to enable v2 enhanced features
        """
        self.mode = mode
        self.fragments = initial_fragments or []
        self.system_time = 0.0
        self.enable_v2_features = enable_v2_features and V2_AVAILABLE
        
        # Initialize operators
        if self.enable_v2_features:
            self.classifier = TopologicalClassifier()
            self.psi_op = EnhancedPsiOperator(self.classifier)
            self.entropy_op = EnhancedEntropyOperator()
            self.meta_loop = EnhancedMetaFeedbackLoop(self.psi_op, self.entropy_op)
            
            # Initialize adaptive neurons
            self.neurons = []
            self.epistemic_particles = []
            
            # Initialize executable axioms
            self.axioms = [
                AX0_SystemicUncertainty(),
                AX1_OntologicalDifference(), 
                AX6_MinimalEntropy()
            ]
            
            logger.info(f"Enhanced GTMØ System initialized in {self.mode.name} mode with v2 features")
        else:
            # Fallback to basic operators
            logger.info(f"GTMØ System initialized in {self.mode.name} mode (basic mode)")
            self.psi_op = None
            self.entropy_op = None
            self.meta_loop = None
        
        self.genesis_history = []
        self.evolution_history = []
    
    def add_adaptive_neuron(self, neuron_id: str, position: Tuple[int, int, int]) -> bool:
        """Add an adaptive neuron to the system (v2 feature)."""
        if not self.enable_v2_features:
            return False
        
        neuron = AdaptiveGTMONeuron(neuron_id, position)
        self.neurons.append(neuron)
        return True
    
    def add_epistemic_particle(self, content: Any, **kwargs) -> bool:
        """Add an epistemic particle to the system (v2 feature)."""
        if not self.enable_v2_features:
            return False
        
        particle = EpistemicParticle(content=content, **kwargs)
        self.epistemic_particles.append(particle)
        return True
    
    def _handle_genesis(self):
        """Handle creation of new fragments based on universe mode."""
        genesis_event = None
        
        if self.mode == UniverseMode.INDEFINITE_STILLNESS:
            # Extremely rare genesis events
            if random.random() < 1e-6:
                genesis_event = {
                    'type': 'rare_genesis',
                    'fragment': f"Spontaneous genesis event at t={self.system_time:.2f}",
                    'probability': 1e-6
                }
                self.fragments.append(genesis_event['fragment'])
                logger.info(f"STILLNESS: Rare genesis - {genesis_event['fragment']}")
        
        elif self.mode == UniverseMode.ETERNAL_FLUX:
            # Frequent chaotic fragment creation
            genesis_rate = 0.4
            if random.random() < genesis_rate:
                chaos_level = random.uniform(0.5, 1.0)
                genesis_event = {
                    'type': 'flux_genesis',
                    'fragment': f"Chaotic flux particle (chaos={chaos_level:.2f}) at t={self.system_time:.2f}",
                    'probability': genesis_rate,
                    'chaos_level': chaos_level
                }
                self.fragments.append(genesis_event['fragment'])
                logger.info(f"FLUX: Chaotic genesis - {genesis_event['fragment']}")
        
        if genesis_event:
            self.genesis_history.append({
                'time': self.system_time,
                'event': genesis_event
            })
    
    def step(self, iterations: int = 1):
        """Advance the simulation by one time step."""
        self.system_time += 1.0
        logger.info(f"--- Enhanced system step {self.system_time}, Fragments: {len(self.fragments)} ---")
        
        # 1. Handle genesis based on universe mode
        self._handle_genesis()
        
        if not self.fragments:
            logger.info("System is empty. No evolution to perform.")
            return
        
        # 2. Enhanced evolution with v2 features
        if self.enable_v2_features:
            self._enhanced_evolution(iterations)
        else:
            self._basic_evolution(iterations)
        
        # 3. Record evolution state
        self.evolution_history.append({
            'time': self.system_time,
            'fragment_count': len(self.fragments),
            'neuron_count': len(self.neurons) if self.enable_v2_features else 0,
            'particle_count': len(self.epistemic_particles) if self.enable_v2_features else 0
        })
    
    def _enhanced_evolution(self, iterations: int):
        """Enhanced evolution using v2 capabilities."""
        # Apply executable axioms
        for axiom in self.axioms:
            axiom.apply(self)
        
        # Evolve epistemic particles
        for particle in self.epistemic_particles:
            particle.evolve(self.system_time / 10.0)
        
        # Run meta-feedback loop on fragments
        if self.meta_loop and self.fragments:
            initial_scores = []
            for fragment in self.fragments:
                try:
                    result = self.psi_op(fragment)
                    score = result.value.get('score', 0.5)
                    initial_scores.append(score)
                except Exception as e:
                    logger.warning(f"Error processing fragment {fragment}: {e}")
                    initial_scores.append(0.5)
            
            feedback_results = self.meta_loop.run(
                self.fragments,
                initial_scores=initial_scores,
                iterations=iterations,
                learning_enabled=True
            )
            
            final_ratios = feedback_results['final_state']['final_classification_ratios']
            logger.info(f"Enhanced evolution step {self.system_time} results: {final_ratios}")
            
            # Store feedback results for analysis
            if not hasattr(self, 'feedback_history'):
                self.feedback_history = []
            self.feedback_history.append({
                'time': self.system_time,
                'results': feedback_results
            })
    
    def _basic_evolution(self, iterations: int):
        """Basic evolution when v2 features not available."""
        logger.info(f"Running basic evolution for {iterations} iterations")
        # Simple placeholder evolution
        pass
    
    def simulate_adversarial_attack(self, attack_type: str, target_indices: List[int], 
                                   intensity: float = 1.0) -> Dict[str, Any]:
        """Simulate adversarial attack on neurons (v2 feature)."""
        if not self.enable_v2_features or not self.neurons:
            return {'error': 'v2 features not available or no neurons'}
        
        results = []
        attack_vector = self._generate_attack_vector(attack_type)
        
        for idx in target_indices:
            if 0 <= idx < len(self.neurons):
                neuron = self.neurons[idx]
                result = neuron.experience_attack(attack_type, attack_vector, intensity)
                results.append({
                    'neuron_id': neuron.id,
                    'result': result,
                    'learned_patterns': neuron.get_learned_patterns()
                })
        
        return {
            'attack_type': attack_type,
            'intensity': intensity,
            'target_count': len(target_indices),
            'results': results,
            'system_time': self.system_time
        }
    
    def _generate_attack_vector(self, attack_type: str) -> Dict[str, float]:
        """Generate attack vector based on type."""
        vectors = {
            'anti_paradox': {'semantic_attack': 0.8, 'logical_attack': 0.9, 'entropy_attack': -0.7},
            'overflow': {'semantic_attack': 2.0, 'logical_attack': 2.0, 'entropy_attack': 2.0},
            'confusion': {'semantic_attack': 0.5, 'logical_attack': -0.5, 'entropy_attack': 0.8},
            'rigid_logic': {'semantic_attack': -0.3, 'logical_attack': -0.9, 'entropy_attack': -0.8}
        }
        
        return vectors.get(attack_type, {'semantic_attack': 0.5, 'logical_attack': 0.5, 'entropy_attack': 0.5})
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        base_report = {
            'system_time': self.system_time,
            'universe_mode': self.mode.name,
            'fragment_count': len(self.fragments),
            'v2_features_enabled': self.enable_v2_features,
            'genesis_events': len(self.genesis_history),
            'evolution_steps': len(self.evolution_history)
        }
        
        if self.enable_v2_features:
            # Enhanced metrics
            base_report.update({
                'neuron_count': len(self.neurons),
                'particle_count': len(self.epistemic_particles),
                'axiom_compliance': {axiom.__class__.__name__: axiom.verify(self) 
                                   for axiom in self.axioms},
                'topological_metrics': self._calculate_topological_metrics(),
                'learning_summary': self._calculate_learning_summary()
            })
            
            # Add feedback history summary
            if hasattr(self, 'feedback_history') and self.feedback_history:
                latest_feedback = self.feedback_history[-1]['results']
                base_report['latest_feedback'] = {
                    'final_ratios': latest_feedback['final_state']['final_classification_ratios'],
                    'adaptation_effectiveness': latest_feedback['final_state'].get('adaptation_effectiveness', 0),
                    'v2_enhanced_metrics': latest_feedback['final_state'].get('v2_enhanced_metrics', {})
                }
        
        return base_report
    
    def _calculate_topological_metrics(self) -> Dict[str, Any]:
        """Calculate topological phase space metrics."""
        if not self.epistemic_particles:
            return {'phase_space_occupied': 0.0, 'attractor_distribution': {}}
        
        # Calculate phase space distribution
        classifications = {}
        for particle in self.epistemic_particles:
            class_type = self.classifier.classify(particle)
            if class_type not in classifications:
                classifications[class_type] = 0
            classifications[class_type] += 1
        
        total = len(self.epistemic_particles)
        distribution = {k.name: v/total for k, v in classifications.items()}
        
        return {
            'phase_space_occupied': len(classifications) / len(KnowledgeType),
            'attractor_distribution': distribution,
            'total_classified': total
        }
    
    def _calculate_learning_summary(self) -> Dict[str, Any]:
        """Calculate learning statistics across all neurons."""
        if not self.neurons:
            return {'total_experiences': 0, 'average_success_rate': 0.0}
        
        total_experiences = sum(n.get_learned_patterns()['total_experiences'] for n in self.neurons)
        success_rates = [n.get_learned_patterns()['success_rate'] for n in self.neurons 
                        if n.get_learned_patterns()['total_experiences'] > 0]
        
        avg_success_rate = np.mean(success_rates) if success_rates else 0.0
        
        return {
            'total_experiences': total_experiences,
            'average_success_rate': avg_success_rate,
            'learning_neurons': len(success_rates),
            'total_neurons': len(self.neurons)
        }


###############################################################################
# Integration and Demonstration
###############################################################################

class EmergenceDetector:
    """Enhanced emergence detector preserving original functionality."""
    
    def __init__(self):
        self.emergence_threshold = 0.8
        self.complexity_threshold = 0.7
        self.novelty_keywords = [
            'emergent', 'novel', 'meta-', 'recursive', 'self-referential',
            'paradox', 'contradiction', 'impossible', 'undefined', 'transcendent',
            'synthesis', 'integration', 'breakthrough'
        ]
    
    def detect_emergence(self, fragment: Any, psi_result: OperationResult, 
                        entropy_result: OperationResult) -> Dict[str, Any]:
        """Enhanced emergence detection with v2 compatibility."""
        emergence_score = 0.0
        emergence_indicators = []
        
        psi_score = psi_result.value.get('score', 0.0)
        total_entropy = entropy_result.value.get('total_entropy', 0.0)
        
        # Enhanced detection with v2 features
        if 0.6 <= psi_score <= 0.9 and 0.3 <= total_entropy <= 0.7:
            emergence_score += 0.3
            emergence_indicators.append('balanced_metrics')
        
        # Check for topological emergence patterns
        if 'phase_point' in psi_result.value:
            phase_point = psi_result.value['phase_point']
            if self._is_phase_space_boundary(phase_point):
                emergence_score += 0.2
                emergence_indicators.append('phase_boundary')
        
        fragment_str = str(fragment).lower()
        novelty_count = sum(1 for keyword in self.novelty_keywords if keyword in fragment_str)
        
        if novelty_count > 0:
            emergence_score += min(0.4, novelty_count * 0.1)
            emergence_indicators.append(f'novelty_keywords_{novelty_count}')
        
        if any(indicator in fragment_str for indicator in ['meta-', 'about itself', 'self-', 'recursive']):
            emergence_score += 0.2
            emergence_indicators.append('meta_cognitive')
        
        if total_entropy > 0.6 and psi_score > 0.7:
            emergence_score += 0.2
            emergence_indicators.append('paradoxical_properties')
        
        is_emergent = emergence_score >= self.emergence_threshold
        emergent_type = None
        
        if is_emergent:
            if 'meta_cognitive' in emergence_indicators:
                emergent_type = 'Ψᴹ (meta-cognitive)'
            elif 'paradoxical_properties' in emergence_indicators:
                emergent_type = 'Ψᴾ (paradoxical)'
            elif 'phase_boundary' in emergence_indicators:
                emergent_type = 'Ψᵀ (topological)'
            elif novelty_count >= 2:
                emergent_type = 'Ψᴺ (novel)'
            else:
                emergent_type = 'Ψᴱ (emergent)'
        
        return {
            'is_emergent': is_emergent,
            'emergence_score': emergence_score,
            'emergent_type': emergent_type,
            'indicators': emergence_indicators,
            'analysis': {
                'psi_score': psi_score,
                'entropy': total_entropy,
                'novelty_count': novelty_count,
                'fragment_length': len(str(fragment)),
                'v2_enhanced': 'phase_point' in psi_result.value
            }
        }
    
    def _is_phase_space_boundary(self, phase_point: Tuple[float, float, float]) -> bool:
        """Check if phase point is near phase space boundaries."""
        boundary_threshold = 0.1
        return any(coord < boundary_threshold or coord > (1 - boundary_threshold) 
                  for coord in phase_point)


def create_enhanced_gtmo_system(mode: UniverseMode = UniverseMode.INDEFINITE_STILLNESS,
                               initial_fragments: Optional[List[Any]] = None,
                               enable_v2: bool = True) -> Tuple[Any, ...]:
    """Factory function to create enhanced GTMØ system with all components."""
    system = EnhancedGTMOSystem(mode, initial_fragments, enable_v2)
    
    if enable_v2 and V2_AVAILABLE:
        return system, system.psi_op, system.entropy_op, system.meta_loop
    else:
        return system, None, None, None


def demonstrate_enhanced_integration():
    """Demonstrate the enhanced integration of v2 features with original functionality."""
    print("=" * 80)
    print("ENHANCED GTMØ SYSTEM - V2 INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Test with both universe modes
    for mode in [UniverseMode.INDEFINITE_STILLNESS, UniverseMode.ETERNAL_FLUX]:
        print(f"\n### TESTING {mode.name} MODE ###")
        print("-" * 50)
        
        # Create enhanced system
        system, psi_op, entropy_op, meta_loop = create_enhanced_gtmo_system(
            mode=mode,
            initial_fragments=["Initial knowledge fragment"],
            enable_v2=V2_AVAILABLE
        )
        
        if V2_AVAILABLE:
            # Add some adaptive neurons
            for i in range(3):
                system.add_adaptive_neuron(f"neuron_{i}", (i, 0, 0))
            
            # Add epistemic particles
            system.add_epistemic_particle("Mathematical theorem", determinacy=0.9, stability=0.9, entropy=0.1)
            system.add_epistemic_particle("Uncertain prediction", determinacy=0.3, stability=0.4, entropy=0.7)
            
            # Test enhanced operators with context-aware AlienatedNumber
            alien_btc = AlienatedNumber("bitcoin_2030", context={
                'temporal_distance': 5.0,
                'volatility': 0.9,
                'predictability': 0.1,
                'domain': 'future_prediction'
            })
            
            psi_result = psi_op(alien_btc)
            entropy_result = entropy_op(alien_btc)
            
            print(f"Context-aware AlienatedNumber processing:")
            print(f"  PSI Score: {psi_result.value['score']:.4f}")
            print(f"  Entropy: {entropy_result.value['total_entropy']:.4f}")
            print(f"  Context factors: {entropy_result.value.get('context_factors', {})}")
            
            # Simulate system evolution
            print(f"\nSimulating 3 evolution steps...")
            for i in range(3):
                system.step(iterations=2)
                report = system.get_comprehensive_report()
                print(f"  Step {i+1}: {report['fragment_count']} fragments, "
                      f"{report.get('neuron_count', 0)} neurons, "
                      f"{report.get('genesis_events', 0)} genesis events")
            
            # Test adversarial attack simulation
            if system.neurons:
                print(f"\nTesting adversarial attack simulation...")
                attack_result = system.simulate_adversarial_attack('anti_paradox', [0, 1], intensity=0.8)
                print(f"  Attack type: {attack_result['attack_type']}")
                print(f"  Targets: {attack_result['target_count']}")
                for result in attack_result['results']:
                    print(f"    Neuron {result['neuron_id']}: "
                          f"Defense={result['result']['defense_used']}, "
                          f"Success={result['result']['success']:.3f}")
        
        else:
            print("v2 features not available - running basic mode")
            for i in range(3):
                system.step()
        
        # Final report
        final_report = system.get_comprehensive_report()
        print(f"\nFinal system state:")
        print(f"  Mode: {final_report['universe_mode']}")
        print(f"  Fragments: {final_report['fragment_count']}")
        print(f"  V2 enabled: {final_report['v2_features_enabled']}")
        print(f"  Genesis events: {final_report['genesis_events']}")
        
        if V2_AVAILABLE and 'latest_feedback' in final_report:
            print(f"  Latest feedback ratios: {final_report['latest_feedback']['final_ratios']}")
    
    print("\n" + "=" * 80)
    print("INTEGRATION SUMMARY:")
    print("✓ Preserved original UniverseMode functionality")
    print("✓ Enhanced with v2 dynamic context-aware calculations")
    print("✓ Integrated topological classification")
    print("✓ Added adaptive learning capabilities")
    print("✓ Maintained backward compatibility")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_enhanced_integration()
