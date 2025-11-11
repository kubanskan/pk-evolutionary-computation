from typing import List, Tuple, Callable
import numpy as np
from real_chromosome import RealIndividual, RealChromosome


class RealPopulation:
    """
    Klasa reprezentująca populację osobników z reprezentacją rzeczywistą.
    """

    def __init__(self,
                 population_size: int,
                 n_variables: int,
                 bounds: List[Tuple[float, float]]):
        """
        Inicjalizacja populacji.

        Args:
            population_size: Wielkość populacji (liczba osobników)
            n_variables: Liczba zmiennych (wymiar problemu)
            bounds: Granice dla każdej zmiennej
        """
        self.population_size = population_size
        self.n_variables = n_variables
        self.bounds = bounds

        self.individuals = [
            RealIndividual(RealChromosome(n_variables, bounds))
            for _ in range(population_size)
        ]

    def __len__(self) -> int:
        return self.population_size

    def __getitem__(self, index: int) -> RealIndividual:
        return self.individuals[index]

    def evaluate(self, fitness_function: Callable[[np.ndarray], float]):

        for individual in self.individuals:
            phenotype = individual.get_phenotype()
            individual.fitness = fitness_function(phenotype)