from .chromosome import BinaryChromosome
import numpy as np


class Mutation:
    """
    Klasa zawierająca metody mutacji dla algorytmu genetycznego.

    Dostępne metody:
    - boundary: Mutacja brzegowa
    - one_point: Mutacja jednopunktowa
    - two_point: Mutacja dwupunktowa

    Parametr konfiguracyjny:
    - mutation_probability: prawdopodobieństwo mutacji
    """

    @staticmethod
    def boundary(chromosome: BinaryChromosome,
                 mutation_probability: float = 0.01) -> BinaryChromosome:
        """
        Mutacja brzegowa
        Zastępuje brzegowe geny z prawdopodobieństwem mutation_probability

        """
        mutated = chromosome.copy()

        if np.random.rand() < mutation_probability:
            point = np.random.randint(0, 2)
            mutated.genes[-point] = 1 - mutated.genes[point]

        return mutated

    @staticmethod
    def one_point(chromosome: BinaryChromosome,
                  mutation_probability: float = 0.01) -> BinaryChromosome:
        """
        Mutacja jednopunktowa

        Z prawdopodobieństwem mutation_probability wybiera JEDEN losowy bit
        w chromosomie i odwraca jego wartość
        """
        mutated = chromosome.copy()

        if np.random.rand() < mutation_probability:
            point = np.random.randint(0, len(mutated.genes))
            mutated.genes[point] = 1 - mutated.genes[point]

        return mutated

    @staticmethod
    def two_point(chromosome: BinaryChromosome,
                  mutation_probability: float = 0.01) -> BinaryChromosome:
        """
        Mutacja dwupunktowa (two-point mutation).

        Z prawdopodobieństwem mutation_probability wybiera DWA losowe bity
        w chromosomie i odwraca ich wartośc
        """
        mutated = chromosome.copy()

        if np.random.rand() < mutation_probability:
            length = len(mutated.genes)
            if length < 2:
                point = np.random.randint(0, length)
                mutated.genes[point] = 1 - mutated.genes[point]
            else:

                points = np.random.choice(length, size=2, replace=False)
                for point in points:
                    mutated.genes[point] = 1 - mutated.genes[point]

        return mutated
