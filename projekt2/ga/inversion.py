from .chromosome import BinaryChromosome
import numpy as np


class Inversion:
    """
    Klasa implementująca operator inwersji (Inversion Mutation)
    Inwersja polega na odwróceniu kolejności genów w losowo wybranym fragmencie chromosomu (pomiędzy dwoma punktami).
    """

    @staticmethod
    def inverse(chromosome: BinaryChromosome,
                inversion_probability: float = 0.1) -> BinaryChromosome:
        """
        Wykonuje mutację inwersji na podanym chromosomie.
        Z prawdopodobieństwem mutation_probability wybiera losowy fragment
        chromosomu i odwraca jego kolejność.

        Returns:
            Nowy chromosom po ewentualnej inwersji
        """

        if np.random.rand() > inversion_probability:
            return chromosome.copy()

        child = chromosome.copy()
        length = len(child.genes)

        point1 = np.random.randint(0, length - 1)
        point2 = np.random.randint(point1 + 1, length)

        child.genes[point1:point2] = child.genes[point1:point2][::-1]

        return child
