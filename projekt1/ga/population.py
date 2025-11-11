from typing import List, Tuple, Callable
from .chromosome import Individual, BinaryChromosome
import numpy as np


class Population:
    """
    Klasa reprezentująca populację osobników.
    """

    def __init__(self,
                 population_size: int,
                 n_variables: int,
                 bounds: List[Tuple[float, float]],
                 precision: int = 6):
        """
        Inicjalizacja populacji.
        Args:
            population_size: Wielkość populacji (liczba osobników)
            n_variables: Liczba zmiennych (wymiar problemu)
            bounds: Granice dla każdej zmiennej
            precision: Dokładność reprezentacji
        """
        self.population_size = population_size
        self.n_variables = n_variables
        self.bounds = bounds
        self.precision = precision

        self.individuals = [
            Individual(BinaryChromosome(n_variables, bounds, precision))
            for _ in range(population_size)
        ]

    def __len__(self) -> int:
        """Zwraca wielkość populacji."""
        return self.population_size

    def __getitem__(self, index: int) -> Individual:
        """Dostęp do osobnika: population[index]."""
        return self.individuals[index]

    def evaluate(self, fitness_function: Callable[[np.ndarray], float]):
        """Ewaluacja wszystkich osobników."""
        for individual in self.individuals:
            phenotype = individual.get_phenotype()
            individual.fitness = fitness_function(phenotype)
