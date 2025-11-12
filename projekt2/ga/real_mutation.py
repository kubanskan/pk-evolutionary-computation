import numpy as np
from .real_chromosome import RealChromosome


class RealMutation:

    @staticmethod
    def uniform(chromosome: RealChromosome,
                mutation_probability: float = 0.1,
                mutation_range: float = 0.1) -> RealChromosome:
        """
        Mutacja równomierna
        Losowy gen przyjmuje wartość z dopuszczalnego przedziału

        """
        mutated = chromosome.copy()

        if np.random.rand() < mutation_probability:
            gene_idx = np.random.randint(0, len(mutated.genes))

            lower, upper = mutated.bounds[gene_idx]
            var_range = upper - lower

            delta = np.random.uniform(-mutation_range * var_range,
                                      mutation_range * var_range)

            mutated.genes[gene_idx] += delta

        mutated.clip_to_bounds()
        return mutated

    @staticmethod
    def gaussian(chromosome: RealChromosome,
                 mutation_probability: float = 0.1,
                 sigma: float = 0.1) -> RealChromosome:
        """
        Mutacja Gaussa
        Zmiana wartości genu w chromosomie poprzez dodanie czynnika jakim
        jest wylosowana liczba z rozkładu normalnego
        """
        mutated = chromosome.copy()

        if np.random.rand() < mutation_probability:
            gene_idx = np.random.randint(0, len(mutated.genes))

            lower, upper = mutated.bounds[gene_idx]
            var_range = upper - lower

            std_dev = sigma * var_range

            delta = np.random.normal(0, std_dev)
            mutated.genes[gene_idx] += delta

        mutated.clip_to_bounds()
        return mutated

