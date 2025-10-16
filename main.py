
from typing import List, Tuple
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

    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.6f}" if self.fitness is not None else "None"
        return f"Individual(fitness={fitness_str}, phenotype={self.get_phenotype()})"


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
        population_size=10,
        n_variables=2,
        bounds=[(-5, 5), (-5, 5)],
        precision=3
    )
    print(f"Wielkość populacji: {len(population)}")

    print(f"\nPierwsze 5 osobników:")
    for i in range(5):
        ind = population[i]
        phenotype = ind.get_phenotype()
        print(ind.chromosome.genes)
        print(f"Osobnik {i}: x={phenotype[0]:.4f}, y={phenotype[1]:.4f}")