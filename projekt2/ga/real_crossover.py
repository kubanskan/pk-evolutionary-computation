from typing import Tuple, List
import numpy as np
from .real_chromosome import RealChromosome


class RealCrossover:
    """
    Klasa zawierająca metody krzyżowania dla reprezentacji rzeczywistej.

    Dostępne metody:
    - arithmetic: Krzyżowanie arytmetyczne
    - linear: Krzyżowanie liniowe (tworzy 3 potomków i wybiera 2 najlepsze)
    - blend_alpha: Krzyżowanie mieszające typu alfa (BLX-α)
    - blend_alpha_beta: Krzyżowanie mieszające typu alfa i beta
    - averaging: Krzyżowanie uśredniające
    """

    @staticmethod
    def arithmetic(parent1: RealChromosome,
                   parent2: RealChromosome,
                   crossover_probability: float = 0.8,
                   alpha: float = 0.5) -> Tuple[RealChromosome, RealChromosome]:
        """
        Krzyżowanie arytmetyczne.

        Tworzy potomków jako kombinację liniową rodziców:
        child1 = α * parent1 + (1 - α) * parent2
        child2 = (1 - α) * parent1 + α * parent2
        """
        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1.genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2.genes = (1 - alpha) * parent1.genes + alpha * parent2.genes

        child1.clip_to_bounds()
        child2.clip_to_bounds()

        return child1, child2

    @staticmethod
    def multi_parent_arithmetic(parents: List[RealChromosome],
                                alphas: Tuple[float, ...] = None) -> RealChromosome:
        """
        Wersja wieloosobnicza krzyżowania arytmetycznego.

        Nowy potomek jest ważoną sumą genów wszystkich rodziców.
        Wymaga, aby suma współczynników alpha wynosiła 1.0 (lub była bliska 1.0).

        Potomek (C) = sum(alpha_i * Parent_i)
        """
        if alphas is None or len(alphas) == 0:
            n_parents = len(parents)
            random_weights = np.random.random(n_parents)
            alphas = tuple(random_weights / random_weights.sum())

        child = parents[0].copy()
        n_genes = len(child.genes)
        new_genes = np.zeros(n_genes)

        for alpha, parent in zip(alphas, parents):
            new_genes += alpha * parent.genes

        child.genes = new_genes
        child.clip_to_bounds()
        return child

    @staticmethod
    def linear(parent1: RealChromosome,
               parent2: RealChromosome,
               crossover_probability: float = 0.8,
               fitness_function=None,
               minimize: bool = True) -> Tuple[RealChromosome, RealChromosome]:
        """
        Krzyżowanie liniowe.

        Tworzy trzech potomków:
        child1 = 0.5 * parent1 + 0.5 * parent2
        child2 = 1.5 * parent1 - 0.5 * parent2
        child3 = -0.5 * parent1 + 1.5 * parent2

        Zwraca dwóch najlepszych
        """
        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent1.copy()
        child3 = parent1.copy()

        child1.genes = 0.5 * parent1.genes + 0.5 * parent2.genes
        child2.genes = 1.5 * parent1.genes - 0.5 * parent2.genes
        child3.genes = -0.5 * parent1.genes + 1.5 * parent2.genes

        child1.clip_to_bounds()
        child2.clip_to_bounds()
        child3.clip_to_bounds()

        children = [child1, child2, child3]
        fitnesses = [fitness_function(child.decode()) for child in children]

        if minimize:
            sorted_indices = np.argsort(fitnesses)
        else:
            sorted_indices = np.argsort(fitnesses)[::-1]

        return children[sorted_indices[0]], children[sorted_indices[1]]

    @staticmethod
    def blend_alpha(parent1: RealChromosome,
                    parent2: RealChromosome,
                    crossover_probability: float = 0.8,
                    alpha: float = 0.5) -> Tuple[RealChromosome, RealChromosome]:
        """
        Krzyżowanie mieszające typu alfa (BLX-α).

        Dla każdego genu:
        - Znajduje min i max wartość wśród rodziców
        - Rozszerza zakres o α * (max - min) w obie strony
        - Losuje wartości potomków z tego rozszerzonego zakresu
        """
        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(len(parent1.genes)):
            min_val = min(parent1.genes[i], parent2.genes[i])
            max_val = max(parent1.genes[i], parent2.genes[i])
            range_val = max_val - min_val

            lower_bound = min_val - alpha * range_val
            upper_bound = max_val + alpha * range_val

            lower, upper = parent1.bounds[i]
            lower_bound = max(lower_bound, lower)
            upper_bound = min(upper_bound, upper)

            child1.genes[i] = np.random.uniform(lower_bound, upper_bound)
            child2.genes[i] = np.random.uniform(lower_bound, upper_bound)

        child1.clip_to_bounds()
        child2.clip_to_bounds()

        return child1, child2

    @staticmethod
    def blend_alpha_beta(parent1: RealChromosome,
                         parent2: RealChromosome,
                         crossover_probability: float = 0.8,
                         alpha: float = 0.5,
                         beta: float = 0.5) -> Tuple[RealChromosome, RealChromosome]:

        """
        Krzyżowanie mieszające typu alfa i beta.

        Podobne do BLX-α, ale rozszerza zakres asymetrycznie:
        - W lewo o α * (max - min)
        - W prawo o β * (max - min)

        """
        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(len(parent1.genes)):
            min_val = min(parent1.genes[i], parent2.genes[i])
            max_val = max(parent1.genes[i], parent2.genes[i])
            range_val = max_val - min_val

            lower_bound = min_val - alpha * range_val
            upper_bound = max_val + beta * range_val

            lower, upper = parent1.bounds[i]
            lower_bound = max(lower_bound, lower)
            upper_bound = min(upper_bound, upper)

            child1.genes[i] = np.random.uniform(lower_bound, upper_bound)
            child2.genes[i] = np.random.uniform(lower_bound, upper_bound)

        child1.clip_to_bounds()
        child2.clip_to_bounds()

        return child1, child2

    @staticmethod
    def averaging(parent1: RealChromosome,
                  parent2: RealChromosome,
                  crossover_probability: float = 0.8) -> Tuple[RealChromosome, RealChromosome]:
        """
        Krzyżowanie uśredniające.

        Tworzy potomków jako średnią arytmetyczną rodziców:
        child1 = child2 = (parent1 + parent2) / 2

        Oba potomki są identyczne (pełna eksploracja może wymagać mutacji).

        """
        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        avg_genes = (parent1.genes + parent2.genes) / 2.0

        child1.genes = avg_genes.copy()
        child2.genes = avg_genes.copy()

        child1.clip_to_bounds()
        child2.clip_to_bounds()

        return child1, child2