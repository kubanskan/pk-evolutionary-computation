from typing import List, Tuple
import numpy as np


class RealChromosome:
    """
    Klasa reprezentująca chromosom rzeczywisty dla algorytmu genetycznego.
    Przechowuje wartości zmiennych bezpośrednio jako liczby rzeczywiste.
    """

    def __init__(self, n_variables: int, bounds: List[Tuple[float, float]]):
        self.n_variables = n_variables
        self.bounds = bounds

        self.genes = np.array([
            np.random.uniform(lower, upper)
            for lower, upper in bounds
        ])

    def decode(self) -> np.ndarray:
        return self.genes.copy()

    def copy(self) -> 'RealChromosome':
        new_chromosome = RealChromosome(self.n_variables, self.bounds)
        new_chromosome.genes = self.genes.copy()
        return new_chromosome

    def clip_to_bounds(self):
        for i in range(self.n_variables):
            lower, upper = self.bounds[i]
            self.genes[i] = np.clip(self.genes[i], lower, upper)


class RealIndividual:
    """
    Klasa reprezentująca osobnika z chromosomem rzeczywistym (chromosom + fitness).
    """

    def __init__(self, chromosome: RealChromosome):
        self.chromosome = chromosome
        self.fitness = None

    def get_phenotype(self) -> np.ndarray:
        return self.chromosome.decode()
