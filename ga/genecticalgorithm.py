from typing import List, Tuple


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
