from typing import Tuple
from dataclasses import dataclass

@dataclass
class GAConfig:
    """Konfiguracja algorytmu genetycznego"""
    population_size: int = 100
    num_generations: int = 100
    num_variables: int = 10
    precision: int = 16
    crossover_prob: float = 0.8
    mutation_prob: float = 0.01
    inversion_prob: float = 0.05
    elite_size: int = 2
    selection_method: str = "tournament"
    tournament_size: int = 3
    selection_percentage: float = 0.5
    crossover_method: str = "one_point"
    mutation_method: str = "one_point"
    bounds: Tuple[float, float] = (-100, 100)
    optimization_type: str = "minimize"