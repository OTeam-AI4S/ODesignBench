"""
Evaluation module for designbench.

Provides task-specific evaluators that assume standardized input format.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.motif_scaffolding.motif_bench import MotifBenchEvaluator

# Factory function to get evaluator by task type
def get_evaluator(task_type: str, config) -> "MotifBenchEvaluator":
    """
    Factory function to get evaluator by task type.
    
    Args:
        task_type: Type of evaluation task (e.g., "motif_scaffolding", "pbp")
        config: Configuration object
        
    Returns:
        Appropriate evaluator instance
    """
    # Lazy import avoids circular import during package initialization.
    from evaluation.motif_scaffolding.motif_bench import MotifBenchEvaluator

    evaluators = {
        "motif_scaffolding": MotifBenchEvaluator,
        "motif_bench": MotifBenchEvaluator,  # Alias
    }
    
    evaluator_class = evaluators.get(task_type)
    if evaluator_class is None:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Available tasks: {list(evaluators.keys())}"
        )
    
    return evaluator_class(config)


def __getattr__(name: str):
    if name == "MotifBenchEvaluator":
        from evaluation.motif_scaffolding.motif_bench import MotifBenchEvaluator
        return MotifBenchEvaluator
    raise AttributeError(f"module 'evaluation' has no attribute '{name}'")

__all__ = [
    'MotifBenchEvaluator',
    'get_evaluator',
]
