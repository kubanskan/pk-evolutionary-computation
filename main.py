import numpy as np
from ga.chromosome import BinaryChromosome, Individual
from ga.population import Population
from ga.elitism import Elitism
from ga.crossover import Crossover
from ga.inversion import Inversion
from ga.mutation import Mutation
from benchmark_functions import Hypersphere

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
        n_variables=4,
        bounds=[(-5, 5), (-5, 5), (0, 1), (-2, 2)],
        precision=3
    )
    print(f"Wielkość populacji: {len(population)}")


    def fitness_function_min(x):
        return np.sum(np.array(x) ** 2)  # minimalizacja


    def fitness_function_max(x: np.ndarray) -> float:
        return np.sum(x)


    population.evaluate(fitness_function_min)

    print("\nPopulacja 4-wymiarowa (minimalizacja):")
    for i, individual in enumerate(population.individuals):
        x, y, z, w = individual.get_phenotype()
        print(f"Osobnik {i + 1}: x={x:.4f}, y={y:.4f}, z={z:.4f}, w={w:.4f}, fitness={individual.fitness:.4f}")

    elites_min = Elitism.elitism_strategy(population, elitism_percentage=0.1, optimization_type='minimize')

    print("\n--- Elitarne osobniki (10% najlepszych, minimalizacja) ---")
    for i, elite in enumerate(elites_min):
        print(f"Elita {i + 1}: fenotyp={elite.get_phenotype()}, fitness={elite.fitness:.4f}")

    # Maksymalizacja
    population.evaluate(fitness_function_max)

    print("\nPopulacja 4-wymiarowa (maksymalizacja):")
    for i, individual in enumerate(population.individuals):
        x, y, z, w = individual.get_phenotype()
        print(f"Osobnik {i + 1}: x={x:.4f}, y={y:.4f}, z={z:.4f}, w={w:.4f}, fitness={individual.fitness:.4f}")

    elites_max = Elitism.elitism_strategy(population, elitism_percentage=0.1, optimization_type='maximize')

    print("\n--- Elitarne osobniki (10% najlepszych, maksymalizacja) ---")
    for i, elite in enumerate(elites_max):
        print(f"Elita {i + 1}: fenotyp={elite.get_phenotype()}, fitness={elite.fitness:.4f}")

    # Test Crossover
    print("\nTest krzyżowania ziarnistego:")

    # Tworzymy dwóch rodziców
    parent1 = BinaryChromosome(n_variables=2, bounds=[(-5, 5), (-5, 5)], precision=3)
    parent2 = BinaryChromosome(n_variables=2, bounds=[(-5, 5), (-5, 5)], precision=3)

    print("Rodzic 1 geny: ", parent1.genes)
    print("Rodzic 2 geny: ", parent2.genes)

    # Krzyżowanie
    child1, child2 = Crossover.discrete(parent1, parent2, crossover_probability=1.0)

    print("\nPotomek 1 geny:", child1.genes)
    print("Potomek 2 geny:", child2.genes)

    # Mutacja
    probabilities = [0.2, 0.5]
    print(f"Chromosom do testu Mutacji {child1.genes}")
    for prob in probabilities:
        print(f"Mutacja prawdopodobieństwo:{prob}%\n")
        print("Jednopunktowa:", Mutation.one_point(child1, prob).genes, "\n")
        print("Dwupunktowa:", Mutation.two_point(child1, prob).genes, "\n")
        print("Brzegowa:", Mutation.boundary(child1, prob).genes, "\n")
        print("Brzegowa:", Mutation.boundary(child1, prob).genes, "\n")
        print("Brzegowa:", Mutation.boundary(child1, prob).genes, "\n\n")

    # Test inwersji na potomstwie
    print("\nTest inwersji na child1:")
    print("Geny przed inwersją:", child1.genes)

    inverted_child = Inversion.inverse(child1, inversion_probability=1.0)
    print(f"Prawdopodobieństwo {prob}: {inverted_child.genes}")

    # Maksymalizacja
    population = Population(
        population_size=100,
        n_variables=2,
        bounds=[(-5, 5), (-5, 5)],
        precision=4
    )
    print(f"Wielkość populacji: {len(population)}")
    hyp = Hypersphere()
    population.evaluate(hyp._evaluate)

    print("\nPopulacja 2-wymiarowa (minimalizacja):")
    for i, individual in enumerate(population.individuals):
        x, y = individual.get_phenotype()
        print(f"Osobnik {i + 1}: x={x:.4f}, y={y:.4f}, fitness={individual.fitness:.4f}")

    elites_max = Elitism.elitism_strategy(population, elitism_percentage=0.1, optimization_type='minimize')
    for i, elite in enumerate(elites_max):
        print(f"Elita {i + 1}: fenotyp={elite.get_phenotype()}, fitness={elite.fitness:.4f}")

