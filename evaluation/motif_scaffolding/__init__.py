"""
Motif Scaffolding evaluation module.

This module contains all resources needed for motif scaffolding evaluation,
including analysis tools, scripts, and resources.
"""
from evaluation.motif_scaffolding.motif_bench import MotifBenchEvaluator
from evaluation.motif_scaffolding.motif_scaffolding_evaluation import MotifScaffoldingEvaluation

__all__ = [
    'MotifBenchEvaluator',
    'MotifScaffoldingEvaluation',
]