
from typing import List, Tuple, Callable
import numpy as np


class BinaryChromosome:
    """
    Klasa reprezentująca chromosom binarny dla algorytmu genetycznego.
    Umożliwia kodowanie wartości rzeczywistych jako ciągi binarne.
    """

    def __init__(self, n_variables: int, bounds: List[Tuple[float, float]],
                 precision: int = 6):
        """
        Inicjalizacja chromosomu binarnego.

        Args:
            n_variables: Liczba zmiennych (wymiar problemu)
            bounds: Lista krotek (dolna_granica, górna_granica) dla każdej zmiennej,
                    granice są dopierane na podstawie standardów dla danej funkcji
            precision: Dokładność (liczba miejsc po przecinku), wskazywana przez użytkownika
        """
        self.n_variables = n_variables
        self.bounds = bounds
        self.precision = precision

        self.bits_per_variable = []
        for lower, upper in bounds:
            range_val = upper - lower
            possible_values = range_val * (10 ** precision) + 1
            n_bits = int(np.ceil(np.log2(possible_values)))
            self.bits_per_variable.append(max(n_bits, 1))

        self.total_length = sum(self.bits_per_variable)

        # Inicjalizacja losowego chromosomu
        self.genes = np.random.randint(0, 2, self.total_length)

    def decode(self) -> np.ndarray:
        """
        Dekodowanie chromosomu binarnego na wartości rzeczywiste.

        Returns:
            Tablica wartości rzeczywistych
        """
        decoded_values = []
        start_idx = 0

        for i, n_bits in enumerate(self.bits_per_variable):
            bits = self.genes[start_idx:start_idx + n_bits]
            start_idx += n_bits

            decimal_value = int(''.join(map(str, bits)), 2)

            max_decimal = 2 ** n_bits - 1

            lower, upper = self.bounds[i]
            if max_decimal > 0:
                real_value = lower + (decimal_value / max_decimal) * (upper - lower)
            else:
                real_value = lower

            decoded_values.append(real_value)

        return np.array(decoded_values)

    def copy(self) -> 'BinaryChromosome':
        """Tworzenie kopii chromosomu."""
        new_chromosome = BinaryChromosome(self.n_variables, self.bounds, self.precision)
        new_chromosome.genes = self.genes.copy()
        return new_chromosome


class Individual:
    """
    Klasa reprezentująca osobnika (chromosom + fitness).
    """

    def __init__(self, chromosome: BinaryChromosome):
        """
        Inicjalizacja osobnika.

        Args:
            chromosome: Chromosom binarny
        """
        self.chromosome = chromosome
        self.fitness = None

    def get_phenotype(self) -> np.ndarray:
        """Zwraca zdekodowane wartości (fenotyp)."""
        return self.chromosome.decode()


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


class GeneticAlgorithmConfig:
    """
    Konfiguracja algorytmu genetycznego.
    """

    def __init__(self,
                 n_variables: int,
                 bounds: List[Tuple[float, float]],
                 precision: int = 6,
                 population_size: int = 100,
                 n_epochs: int = 100):
        """
        Inicjalizacja konfiguracji algorytmu genetycznego.

        Args:
            n_variables: Liczba zmiennych (wymiarowość problemu)
            bounds: Granice dla każdej zmiennej
            precision: Dokładność reprezentacji
            population_size: Wielkość populacji
            n_epochs: Liczba epok (pokoleń)
        """
        self.n_variables = n_variables
        self.bounds = bounds
        self.precision = precision
        self.population_size = population_size
        self.n_epochs = n_epochs


class Selection:
    """
    Klasa zawierająca wszystkie metody selekcji dla algorytmu genetycznego.

    Dostępne metody:
    - best: Selekcja najlepszych (parametr: selection_percentage)
    - roulette: Selekcja kołem ruletki
    - tournament: Selekcja turniejowa (parametr: tournament_size)
    """

    @staticmethod
    def best(population: Population,
             n_parents: int,
             selection_percentage: float = 0.5,
             optimization_type: str = 'minimize') -> List[Individual]:
        """
        Selekcja najlepszych osobników (elitarna).

        Wybiera określony procent najlepszych osobników z populacji.
        Im wyższy procent, tym więcej osobników zostanie wybranych.

        Parametry konfiguracyjne:
        - selection_percentage: procent najlepszych osobników (0.0-1.0)

        Args:
            population: Populacja do selekcji
            n_parents: Maksymalna liczba rodziców
            selection_percentage: Procent najlepszych (domyślnie 50%)
            optimization_type: 'minimize' lub 'maximize'

        Returns:
            Lista wybranych osobników
        """

        n_selected = max(1, int(len(population.individuals) * selection_percentage))
        n_selected = min(n_selected, n_parents)

        if optimization_type == 'minimize':
            sorted_individuals = sorted(population.individuals,
                                        key=lambda ind: ind.fitness)
        else:
            sorted_individuals = sorted(population.individuals,
                                        key=lambda ind: ind.fitness,
                                        reverse=True)

        return sorted_individuals[:n_selected]

    @staticmethod
    def roulette(population: Population,
                 n_parents: int,
                 optimization_type: str = 'minimize') -> List[Individual]:
        """
        Selekcja kołem ruletki (proporcjonalna do fitness).

        Przygotowujemy koło ruletki gdzie:
        - Im osobnik lepszy, tym większy % na kole ruletki
        - Szansa wylosowania proporcjonalna do fitness

        Proces:
        1. Każdy osobnik ma "wycinek" koła proporcjonalny do fitness
        2. Lepsze osobniki = większe wycinki = większa szansa
        3. Losujemy n_parents razy (kręcimy kołem)

        Args:
            population: Populacja do selekcji
            n_parents: Liczba rodziców do wybrania
            optimization_type: 'minimize' lub 'maximize'

        Returns:
            Lista wybranych osobników

        """
        fitnesses = np.array([ind.fitness for ind in population.individuals])

        # Dla minimalizacji: przekształcamy fitness
        if optimization_type == 'minimize':
            min_fitness = np.min(fitnesses)
            if min_fitness < 0:
                fitnesses = fitnesses - min_fitness + 1
            fitnesses = 1.0 / (fitnesses + 1e-10)


        total_fitness = np.sum(fitnesses)
        probabilities = fitnesses / total_fitness

        selected_indices = np.random.choice(
            len(population.individuals),
            size=n_parents,
            replace=True,
            p=probabilities
        )

        return [population.individuals[i] for i in selected_indices]

    @staticmethod
    def tournament(population: Population,
                   n_parents: int,
                   tournament_size: int = 3,
                   optimization_type: str = 'minimize') -> List[Individual]:
        """
        Selekcja turniejowa

        Tworzy n turniejów po k osobników w każdym.
        Z każdego turnieju wybieramy zwycięzcę (najlepszego).

        Parametry konfiguracyjne:
        - tournament_size (k): wielkość turnieju

        Przykład dla k=3:
        1. Losujemy 3 osobników do turnieju
        2. Porównujemy ich fitness
        3. Wybieramy najlepszego jako zwycięzcę
        4. Powtarzamy n_parents razy

        Args:
            population: Populacja do selekcji
            n_parents: Liczba rodziców (liczba turniejów)
            tournament_size: Rozmiar turnieju k (domyślnie 3)
            optimization_type: 'minimize' lub 'maximize'

        Returns:
            Lista zwycięzców turniejów
        """

        selected = []


        for _ in range(n_parents):
            tournament_indices = np.random.choice(
                len(population.individuals),
                size=min(tournament_size, len(population.individuals)),
                replace=False
            )
            tournament = [population.individuals[i] for i in tournament_indices]

            if optimization_type == 'minimize':
                winner = min(tournament, key=lambda ind: ind.fitness)
            else:
                winner = max(tournament, key=lambda ind: ind.fitness)

            selected.append(winner)

        return selected

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

        print(a)
        child1_genes = np.where(a <= 0.5, parent1.genes, parent2.genes)
        child2_genes = np.where(a <= 0.5, parent2.genes, parent1.genes)

        child1 = parent1.copy()
        child2 = parent2.copy()
        child1.genes = child1_genes
        child2.genes = child2_genes

        return child1, child2


if __name__ == "__main__":

    # Test Binary Chromosome
    bounds = [(-5.0, 5.0), (-10.0, 10.0)]
    chromosome = BinaryChromosome(n_variables=2, bounds=bounds, precision=4)
    print("Geny chromosomu:", chromosome.genes)
    decoded = chromosome.decode()
    print("Zdekodowane wartości:", decoded)


    # Test Individual
    print("\nTest Individual:")
    chromosome = BinaryChromosome(n_variables=2, bounds=[(-5, 5), (-5, 5)], precision=4)
    individual = Individual(chromosome)
    print(f"Fenotyp: {individual.get_phenotype()}")
    print(f"Fitness: {individual.fitness}")

    # Ustawienie fitness
    individual.fitness = 12.345
    print(f"Po ustawieniu fitness: {individual}")

    # Test Population
    print("\nTest Population:")
    population = Population(
        population_size=5,
        n_variables=4,
        bounds=[(-5, 5), (-5, 5), (0, 1), (-2, 2)],
        precision=3
    )
    print(f"Wielkość populacji: {len(population)}")

    print("\nPopulacja 4-wymiarowa:")
    for i, individual in enumerate(population.individuals):
        x, y, z, w = individual.get_phenotype()
        print(f"Osobnik {i + 1}: x={x:.4f}, y={y:.4f}, z={z:.4f}, w={w:.4f}")

    # Test Crossover
    print("\nTest krzyżowania jednopunktowego:")

    # Tworzymy dwóch rodziców
    parent1 = BinaryChromosome(n_variables=2, bounds=[(-5, 5), (-5, 5)], precision=3)
    parent2 = BinaryChromosome(n_variables=2, bounds=[(-5, 5), (-5, 5)], precision=3)

    print("Rodzic 1 geny: ", parent1.genes)
    print("Rodzic 2 geny: ", parent2.genes)

    # Krzyżowanie
    child1, child2 = Crossover.discrete(parent1, parent2, crossover_probability=1.0)

    print("\nPotomek 1 geny:", child1.genes)
    print("Potomek 2 geny:", child2.genes)
