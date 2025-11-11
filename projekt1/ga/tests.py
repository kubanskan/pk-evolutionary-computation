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
    print("=== Test Binary Chromosome ===")
    bounds = [(-5.0, 5.0), (-10.0, 10.0)]
    chromosome = BinaryChromosome(n_variables=2, bounds=bounds, precision=4)
    print("Geny chromosomu:", chromosome.genes)
    print("Zdekodowane wartości:", chromosome.decode())

    # Test Individual
    print("\n=== Test Individual ===")
    individual = Individual(chromosome)
    print(f"Fenotyp: {individual.get_phenotype()}")
    print(f"Fitness: {individual.fitness}")
    individual.fitness = 12.345
    print(f"Po ustawieniu fitness: {individual.fitness}")

    # Test Population
    print("\n=== Test Population ===")
    population = Population(
        population_size=10,
        n_variables=2,
        bounds=[(-5, 5), (-5, 5)],
        precision=3
    )
    print(f"Wielkość populacji: {len(population)}")

    # Test Elitism
    print("\n=== Test Elitism ===")
    fitness_func = lambda x: np.sum(np.array(x) ** 2)
    population.evaluate(fitness_func)

    for i in range(3):
        x, y = population.individuals[i].get_phenotype()
        print(f"Osobnik {i + 1}: x={x:.4f}, y={y:.4f}, fitness={population.individuals[i].fitness:.4f}")

    elites = Elitism.elitism_strategy(population, elitism_percentage=0.2, optimization_type='minimize')
    print(f"Liczba elit (20%): {len(elites)}")

    # Test Crossover
    print("\n=== Test Crossover ===")
    parent1 = BinaryChromosome(n_variables=2, bounds=[(-5, 5), (-5, 5)], precision=3)
    parent2 = BinaryChromosome(n_variables=2, bounds=[(-5, 5), (-5, 5)], precision=3)
    print("Rodzic 1:", parent1.genes)
    print("Rodzic 2:", parent2.genes)

    child1, child2 = Crossover.discrete(parent1, parent2, crossover_probability=1.0)
    print("Potomek 1:", child1.genes)
    print("Potomek 2:", child2.genes)

    # Test Mutation
    print("\n=== Test Mutation ===")
    print("Oryginalny:", child1.genes)
    print("Jednopunktowa:", Mutation.one_point(child1, 0.3).genes)
    print("Dwupunktowa:", Mutation.two_point(child1, 0.3).genes)
    print("Brzegowa:", Mutation.boundary(child1, 0.3).genes)

    # Test Inversion
    print("\n=== Test Inversion ===")
    print("Przed inwersją:", child1.genes)
    inverted = Inversion.inverse(child1, inversion_probability=1.0)
    print("Po inwersji:", inverted.genes)

    # Test z Hypersphere
    print("\n=== Test Hypersphere ===")
    population = Population(
        population_size=5,
        n_variables=2,
        bounds=[(-5, 5), (-5, 5)],
        precision=4
    )
    hyp = Hypersphere()
    population.evaluate(hyp)

    for i in range(5):
        x, y = population.individuals[i].get_phenotype()
        print(f"Osobnik {i + 1}: x={x:.4f}, y={y:.4f}, fitness={population.individuals[i].fitness:.4f}")

    print("\n✓ Wszystkie testy zakończone")