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
