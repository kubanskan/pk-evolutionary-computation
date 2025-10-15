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
                    granice są dopierane na podsawie standardów dla danej funkckji
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


class Population:
    """
    Klasa reprezentująca populację chromosomów.
    Implementacja konfiguracji wielkości populacji.
    """

    def __init__(self,
                 population_size: int,
                 n_variables: int,
                 bounds: List[Tuple[float, float]],
                 precision: int = 6):
        """
        Inicjalizacja populacji.
        Args:
            population_size: Wielkość populacji (liczba chromosomów)
            n_variables: Liczba zmiennych (wymiar problemu)
            bounds: Granice dla każdej zmiennej
            precision: Dokładność reprezentacji
        """
        self.population_size = population_size
        self.n_variables = n_variables
        self.bounds = bounds
        self.precision = precision

        self.chromosomes = [
            BinaryChromosome(n_variables, bounds, precision)
            for _ in range(population_size)
        ]

    def __len__(self) -> int:
        """Zwraca wielkość populacji."""
        return self.population_size

    def __getitem__(self, index: int) -> BinaryChromosome:
        """Dostęp do chromosomu: population[index]."""
        return self.chromosomes[index]


if __name__ == "__main__":

    # Test Binary Chromosome
    bounds = [(-5.0, 5.0), (-10.0, 10.0)]
    chromosome = BinaryChromosome(n_variables=2, bounds=bounds, precision=4)
    print("Geny chromosomu:", chromosome.genes)
    decoded = chromosome.decode()
    print("Zdekodowane wartości:", decoded)

    # Test Population
    population = Population(
        population_size=10,
        n_variables=2,
        bounds=[(-5, 5), (-5, 5)],
        precision=3
    )
    print(f"Wielkość populacji: {len(population)}")
    print(f"\nPierwsze 5 chromosomów:")

    for i in range(5):
        decoded = population[i].decode()
        print(f"Chromosom {i}: x={decoded[0]:.4f}, y={decoded[1]:.4f}")
        print("Geny:", population[i].genes)
