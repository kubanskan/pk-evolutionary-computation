from .chromosome import BinaryChromosome
from typing import Tuple
import numpy as np


class Crossover:
    """
    Klasa zawierająca wszystkie metody krzyżowania dla algorytmu genetycznego.

    Dostępne metody:
    - singlePoint: Krzyżowanie jednopunktowe
    - twoPoint: Krzyżowanie dwupunktowe
    - uniform: Krzyżowanie jednorodne
    - discrete: Krzyżowanie ziarniste
    """

    @staticmethod
    def one_point(parent1: BinaryChromosome,
                  parent2: BinaryChromosome,
                  crossover_probability: float = 0.8) -> Tuple[BinaryChromosome, BinaryChromosome]:
        """
        Krzyżowanie jednopunktowe.

        Wybiera losowy punkt podziału genów rodziców i wymienia fragmenty
        chromosomów po tym punkcie, tworząc dwóch potomków.

        Parametry konfiguracyjne:
        - crossover_probability: Prawdopodobieństwo wykonania krzyżowania (0-1)

        Args:
            parent1: Pierwszy rodzic (BinaryChromosome)
            parent2: Drugi rodzic (BinaryChromosome)
            crossover_probability: Prawdopodobieństwo krzyżowania

        Returns:
            Krotka (child1, child2) – potomkowie po krzyżowaniu
        """

        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        point = np.random.randint(1, len(parent1.genes))

        child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])

        child1 = parent1.copy()
        child2 = parent2.copy()
        child1.genes = child1_genes
        child2.genes = child2_genes

        return child1, child2

    @staticmethod
    def two_point(parent1: BinaryChromosome,
                  parent2: BinaryChromosome,
                  crossover_probability: float = 0.8) -> Tuple[BinaryChromosome, BinaryChromosome]:
        """
        Krzyżowanie dwupunktowe

        Wybiera dwa punkty podziału genów rodziców i wymienia fragment środkowy
        pomiędzy nimi, tworząc dwóch potomków.

        Parametry konfiguracyjne:
        - crossover_probability: Prawdopodobieństwo wykonania krzyżowania (0-1)

        Args:
            parent1: Pierwszy rodzic (BinaryChromosome)
            parent2: Drugi rodzic (BinaryChromosome)
            crossover_probability: Prawdopodobieństwo krzyżowania

        Returns:
            Krotka (child1, child2) – potomkowie po krzyżowaniu
        """
        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        length = len(parent1.genes)
        point1 = np.random.randint(1, length - 1)
        point2 = np.random.randint(point1 + 1, length)

        child1_genes = np.hstack((
            parent1.genes[:point1],
            parent2.genes[point1:point2],
            parent1.genes[point2:]
        ))

        child2_genes = np.hstack((
            parent2.genes[:point1],
            parent1.genes[point1:point2],
            parent2.genes[point2:]
        ))

        child1 = parent1.copy()
        child2 = parent2.copy()
        child1.genes = child1_genes
        child2.genes = child2_genes

        return child1, child2

    @staticmethod
    def uniform(parent1: BinaryChromosome,
                parent2: BinaryChromosome,
                crossover_probability: float = 0.8,
                p: float = 0.5) -> Tuple[BinaryChromosome, BinaryChromosome]:
        """
        Krzyżowanie jednorodne (Uniform Crossover, UX)

        Dla każdego genu losowane jest prawdopodobieństwo α z przedziału [0,1].
        Jeśli α < p (domyślnie 0.5), bity rodziców są wymieniane między sobą.
        Dodatkowo, całe krzyżowanie odbywa się z określonym prawdopodobieństwem
        `crossover_probability` (np. 0.8).

        Parametry konfiguracyjne:
        - crossover_probability: prawdopodobieństwo wykonania całego krzyżowania (0–1)
        - p: prawdopodobieństwo wymiany pojedynczego genu (0–1)

        Args:
            parent1: Pierwszy rodzic (BinaryChromosome)
            parent2: Drugi rodzic (BinaryChromosome)
            crossover_probability: Prawdopodobieństwo wykonania krzyżowania
            p: Prawdopodobieństwo wymiany genu między rodzicami
        """

        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        mask = np.random.rand(len(parent1.genes)) < p

        child1_genes = np.where(mask, parent2.genes, parent1.genes)
        child2_genes = np.where(mask, parent1.genes, parent2.genes)

        child1.genes = child1_genes
        child2.genes = child2_genes

        return child1, child2

    @staticmethod
    def discrete(parent1: BinaryChromosome,
                 parent2: BinaryChromosome,
                 crossover_probability: float = 0.8) -> Tuple[BinaryChromosome, BinaryChromosome]:
        """
        Krzyżowanie ziarniste (Discrete Crossover).

        Dla każdego genu losowana jest liczba a z przedziału [0,1].
        Jeśli a <= 0.5, gen w potomstwie pochodzi od rodzica 1, w przeciwnym wypadku od rodzica 2.

        Parametry konfiguracyjne:
        - crossover_probability: Prawdopodobieństwo wykonania krzyżowania (0-1)

        Args:
            parent1: Pierwszy rodzic (BinaryChromosome)
            parent2: Drugi rodzic (BinaryChromosome)
            crossover_probability: Prawdopodobieństwo krzyżowania

        Returns:
            Krotka (child1, child2) – potomkowie po krzyżowaniu
        """

        if np.random.rand() > crossover_probability:
            return parent1.copy(), parent2.copy()

        a = np.random.rand(len(parent1.genes))

        child1_genes = np.where(a <= 0.5, parent1.genes, parent2.genes)
        child2_genes = np.where(a <= 0.5, parent2.genes, parent1.genes)

        child1 = parent1.copy()
        child2 = parent2.copy()
        child1.genes = child1_genes
        child2.genes = child2_genes

        return child1, child2
