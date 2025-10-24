from population import Population
from chromosome import Individual
from typing import List
import numpy as np


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
