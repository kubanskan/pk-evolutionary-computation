import time
import numpy as np
from typing import Callable, List
from app.config import GAConfig
from ga.chromosome import Individual
from ga.population import Population
from ga.mutation import Mutation
from ga.crossover import Crossover
from ga.selection import Selection
from ga.elitism import Elitism
from ga.inversion import Inversion


class GeneticAlgorithmConfig:
    """
    Główna klasa algorytmu genetycznego z pełną implementacją ewolucji.
    """

    def __init__(self, config: GAConfig, objective_function: Callable):
        """
        Inicjalizacja algorytmu genetycznego.

        Args:
            config: Konfiguracja parametrów GA
            objective_function: Funkcja celu do optymalizacji
        """
        self.config = config
        self.objective_function = objective_function

        bounds = [(config.bounds[0], config.bounds[1])
                  for _ in range(config.num_variables)]

        self.population = Population(
            population_size=config.population_size,
            n_variables=config.num_variables,
            bounds=bounds,
            precision=config.precision
        )

        self.best_individual = None
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'std_fitness': []
        }

        self.selection_strategy = self.get_selection_strategy()
        self.crossover_strategy = self.get_crossover_strategy()
        self.mutation_strategy = self.get_mutation_strategy()

    def get_selection_strategy(self):
        """Wybór strategii selekcji."""
        strategies = {
            "best": lambda pop, n: Selection.best(
                pop, n,
                selection_percentage=0.5,
                optimization_type=self.config.optimization_type
            ),
            "roulette": lambda pop, n: Selection.roulette(
                pop, n,
                optimization_type=self.config.optimization_type
            ),
            "tournament": lambda pop, n: Selection.tournament(
                pop, n,
                tournament_size=self.config.tournament_size,
                optimization_type=self.config.optimization_type
            )
        }
        return strategies[self.config.selection_method]

    def get_crossover_strategy(self):
        """Wybór strategii krzyżowania."""
        strategies = {
            "one_point": lambda p1, p2: Crossover.one_point(
                p1, p2,
                crossover_probability=self.config.crossover_prob
            ),
            "two_point": lambda p1, p2: Crossover.two_point(
                p1, p2,
                crossover_probability=self.config.crossover_prob
            ),
            "uniform": lambda p1, p2: Crossover.uniform(
                p1, p2,
                crossover_probability=self.config.crossover_prob
            ),
            "discrete": lambda p1, p2: Crossover.discrete(
                p1, p2,
                crossover_probability=self.config.crossover_prob
            )
        }
        return strategies[self.config.crossover_method]

    def get_mutation_strategy(self):
        """Wybór strategii mutacji."""
        strategies = {
            "one_point": lambda c: Mutation.one_point(
                c,
                mutation_probability=self.config.mutation_prob
            ),
            "two_point": lambda c: Mutation.two_point(
                c,
                mutation_probability=self.config.mutation_prob
            ),
            "boundary": lambda c: Mutation.boundary(
                c,
                mutation_probability=self.config.mutation_prob
            )
        }
        return strategies[self.config.mutation_method]

    def fitness_function(self, phenotype: np.ndarray) -> float:
        """
        Wrapper dla funkcji fitness - wywołuje funkcję celu.
        Returns:
            Wartość funkcji celu
        """
        return self.objective_function(phenotype)

    def update_best_individual(self):
        """Aktualizacja najlepszego osobnika w populacji."""
        if self.config.optimization_type == 'minimize':
            current_best = min(self.population.individuals,
                               key=lambda ind: ind.fitness)
        else:
            current_best = max(self.population.individuals,
                               key=lambda ind: ind.fitness)

        if self.best_individual is None:
            self.best_individual = current_best
        else:
            if self.config.optimization_type == 'minimize':
                if current_best.fitness < self.best_individual.fitness:
                    self.best_individual = current_best
            else:
                if current_best.fitness > self.best_individual.fitness:
                    self.best_individual = current_best

    def collect_statistics(self, generation: int):
        """
        Zbieranie statystyk z bieżącej generacji.

        Args:
            generation: Numer generacji
        """
        fitnesses = [ind.fitness for ind in self.population.individuals]

        self.history['generation'].append(generation)
        self.history['best_fitness'].append(self.best_individual.fitness)
        self.history['avg_fitness'].append(np.mean(fitnesses))
        self.history['std_fitness'].append(np.std(fitnesses))

    def create_new_population(self, elites: List[Individual]) -> List[Individual]:
        """
        Tworzenie nowej populacji z wykorzystaniem operatorów genetycznych.
        """
        new_population = elites.copy()
        while len(new_population) < self.config.population_size:

            parents = self.selection_strategy(self.population, 2)

            if len(parents) < 2:
                parents = parents + parents  # Duplikuj jeśli jest tylko jeden

            parent1, parent2 = parents[0], parents[1]

            child1_chromosome, child2_chromosome = self.crossover_strategy(
                parent1.chromosome,
                parent2.chromosome
            )

            child1_chromosome = self.mutation_strategy(child1_chromosome)
            child2_chromosome = self.mutation_strategy(child2_chromosome)

            if np.random.rand() < self.config.inversion_prob:
                child1_chromosome = Inversion.inverse(
                    child1_chromosome,
                    inversion_probability=1.0
                )
            if np.random.rand() < self.config.inversion_prob:
                child2_chromosome = Inversion.inverse(
                    child2_chromosome,
                    inversion_probability=1.0
                )


            child1 = Individual(child1_chromosome)
            child2 = Individual(child2_chromosome)

            if len(new_population) < self.config.population_size:
                new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        return new_population

    def evolve(self) -> dict:
        """
        Główna pętla ewolucyjna algorytmu genetycznego.
        Returns:
            Słownik z wynikami optymalizacji
        """
        start_time = time.time()


        self.population.evaluate(self.fitness_function)
        self.update_best_individual()

        for generation in range(1, self.config.num_generations + 1):
            elites = []
            if self.config.elite_size > 0:
                elites = Elitism.elitism_strategy(
                    self.population,
                    elitism_percentage=self.config.elite_size / self.config.population_size,
                    optimization_type=self.config.optimization_type
                )

            new_individuals = self.create_new_population(elites)

            self.population.individuals = new_individuals
            self.population.evaluate(self.fitness_function)
            self.update_best_individual()
            self.collect_statistics(generation)

        elapsed_time = time.time() - start_time

        result = {
            'best_fitness': self.best_individual.fitness,
            'best_solution': self.best_individual.get_phenotype().tolist(),
            'elapsed_time': elapsed_time,
            'history': self.history,
            'generations': self.config.num_generations,
            'final_population_size': len(self.population.individuals)
        }

        return result