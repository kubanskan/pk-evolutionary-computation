from .population import Population
from typing import List
from .chromosome import Individual


class Elitism:
    """
    Klasa realizująca strategię elitarną w algorytmie genetycznym.

    Strategia elitarna polega na zachowaniu najlepszych osobników
    z bieżącej populacji i przeniesieniu ich do nowej populacji,
    aby uniknąć utraty najlepszego rozwiązania.
    """

    @staticmethod
    def elitism_strategy(population: Population,
                         elitism_percentage: float = 0.1,
                         optimization_type: str = 'minimize') -> List[Individual]:
        """
        Selekcja elitarnych osobników – wybiera najlepiej dopasowanych osobników
        z populacji i zachowuje ich do następnej generacji.

        Args:
            population: Populacja do selekcji
            elitism_percentage: Procent populacji do zachowania (0.0 - 1.0)
            optimization_type: 'minimize' lub 'maximize'

        Returns:
            Lista elitarnych osobników
        """
        pop_size = len(population)
        n_elites = max(1, int(pop_size * elitism_percentage))  # liczba elit

        if optimization_type == 'minimize':
            sorted_individuals = sorted(population.individuals, key=lambda ind: ind.fitness)
        else:
            sorted_individuals = sorted(population.individuals, key=lambda ind: ind.fitness, reverse=True)

        elites = sorted_individuals[:n_elites]
        return elites
