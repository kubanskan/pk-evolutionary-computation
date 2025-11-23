import numpy as np
from projekt2.ga.chromosome import BinaryChromosome, Individual
from projekt2.ga.population import Population
from projekt2.ga.elitism import Elitism
from projekt2.ga.crossover import Crossover
from projekt2.ga.inversion import Inversion
from projekt2.ga.mutation import Mutation
from benchmark_functions import Hypersphere
from projekt2.ga.real_chromosome import RealIndividual, RealChromosome
from projekt2.ga.real_population import RealPopulation
from projekt2.ga.real_mutation import RealMutation
from projekt2.ga.real_crossover import RealCrossover


def polynomial_3rd_degree_fitness(phenotype: np.ndarray) -> float:
    """
    Funkcja celu 3. stopnia: f(x1, x2) = x1^3 + x2^2 - 3*x1 + 4
    Wymaga, aby fenotyp miał co najmniej 2 zmienne.
    """
    if phenotype.shape[0] < 2:
        raise ValueError("Funkcja fitness wymaga co najmniej 2 zmiennych.")

    x1 = phenotype[0]
    x2 = phenotype[1]

    # x1^3 + x2^2 - 3*x1 + 4
    return (x1 ** 3) + (x2 ** 2) - (3 * x1) + 4

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

    print("\n\n======== RZECZYWISTA REPREZENTACJA ========\n")

    # 1. Test RealChromosome
    print("=== Test RealChromosome ===")
    bounds_test = [(-5.0, 5.0), (10.0, 20.0), (-1.0, 1.0)]
    n_vars_test = len(bounds_test)

    real_chromosome = RealChromosome(n_variables=n_vars_test, bounds=bounds_test)
    print(f"Geny chromosomu (Rzeczywiste Wartości): {real_chromosome.genes}")

    real_chromosome.genes[0] = 100.0
    print(f"Zmieniony pierwszy gen chromosomu (Rzeczywiste Wartości): {real_chromosome.genes}")

    real_chromosome.clip_to_bounds()
    print(f"Geny po clip_to_bounds: {real_chromosome.genes}")

    # 2. Test RealIndividual
    print("\n=== Test RealIndividual ===")
    real_individual = RealIndividual(real_chromosome.copy())

    fitness_value = polynomial_3rd_degree_fitness(real_individual.chromosome.decode())
    real_individual.fitness = fitness_value
    print(f"Fenotyp: {real_individual.chromosome.decode()}")
    print(f"Fitness (x1^3 + x2^2 - 3x1 + 4): {real_individual.fitness:.4f}")

    # 3. Test RealPopulation
    print("\n=== Test RealPopulation ===")
    population_size = 5

    real_population = RealPopulation(
        population_size=population_size,
        n_variables=2,
        bounds=bounds_test
    )
    print(f"Wielkość populacji: {len(real_population)}")

    real_population.evaluate(polynomial_3rd_degree_fitness)

    print("\n--- Po ocenie (evaluate) ---")

    for idx, individual in enumerate(real_population.individuals):
        phenotype = individual.chromosome.decode()
        fitness = individual.fitness

        phenotype_str = np.array2string(phenotype, precision=4, suppress_small=True)

        print(f"Osobnik {idx}:")
        print(f"  Fenotyp (x1, x2): {phenotype_str}")
        print(f"  Fitness: {fitness:.4f}")

    best_individual = min(real_population.individuals, key=lambda ind: ind.fitness)

    print(f"Fenotyp Najlepszego Osob.: {best_individual.chromosome.decode()}")
    print(f"Najlepszy Fitness w populacji: {best_individual.fitness:.4f} (blisko 2.0)")

    # Przygotowanie chromosomów do mutacji i krzyżowania (dla 3 zmiennych)
    P1_bounds = [(-5.0, 5.0), (0.0, 10.0), (-10.0, -5.0)]
    P_c = 1.0  # Prawdopodobieństwo krzyżowania
    P_m = 1.0  # Prawdopodobieństwo mutacji

    parent_real1 = RealChromosome(n_variables=3, bounds=P1_bounds)
    parent_real2 = RealChromosome(n_variables=3, bounds=P1_bounds)

    # Ustawienie konkretnych wartości dla weryfikacji
    parent_real1.genes = np.array([-1.5, 3.0, -8.0])
    parent_real2.genes = np.array([2.5, 7.0, -6.0])

    ## 4. Test RealMutation
    print("\n=== 4. Test RealMutation ===")
    print(f"Oryginalny Gen: {parent_real1.genes}")

    # Test Mutacji Równomiernej
    mutated_uniform = RealMutation.uniform(parent_real1, mutation_probability=P_m, mutation_range=0.1)
    print(f"Mutacja Równomierna: {mutated_uniform.genes}")

    # Test Mutacji Gaussa
    mutated_gaussian = RealMutation.gaussian(parent_real1, mutation_probability=P_m, sigma=0.1)
    print(f"Mutacja Gaussa: {mutated_gaussian.genes}")

    ## 5. Test RealCrossover
    print("\n=== 5. Test RealCrossover ===")
    print(f"Rodzic 1: {parent_real1.genes}")
    print(f"Rodzic 2: {parent_real2.genes}")

    # Przygotowanie chromosomów do krzyżowania (2 zmienne dla testu linear)
    P_bounds = [(-2.0, 2.0), (-2.0, 2.0)]  # Granice dla x1 i x2
    P_c = 1.0  # Prawdopodobieństwo krzyżowania

    parent_real1 = RealChromosome(n_variables=2, bounds=P_bounds)
    parent_real2 = RealChromosome(n_variables=2, bounds=P_bounds)


    parent_real1.genes = np.array([1.0, 0.5])
    parent_real2.genes = np.array([-1.0, 1.5])

    print("\n\n======== TESTY REAL CROSSOVER ========\n")
    print("Ustawienie konkretnych, łatwych do weryfikacji wartości\n")

    print(f"Rodzic 1 (P1): {parent_real1.genes}")
    print(f"Rodzic 2 (P2): {parent_real2.genes}")
    print(f"Prawdopodobieństwo krzyżowania (Pc): {P_c}")
    print("-" * 40)

    ## 5. Test RealCrossover

    # 5.1 Krzyżowanie Arytmetyczne (Arithmetic)
    # Child1 = 0.5 * P1 + 0.5 * P2 (dla alpha=0.5)
    # Child2 = 0.5 * P1 + 0.5 * P2 (dla alpha=0.5)
    child_arith1, child_arith2 = RealCrossover.arithmetic(parent_real1, parent_real2, crossover_probability=P_c,
                                                          alpha=0.5)
    print("=== 5.1 Krzyżowanie Arytmetyczne (alpha=0.5) ===")
    print(f"Potomek Arithmetic 1: {child_arith1.genes} (Powinno być [0.0, 1.0])")
    print(f"Potomek Arithmetic 2: {child_arith2.genes} (Powinno być [0.0, 1.0])")
    print("-" * 40)

    # 5.2 Krzyżowanie Liniowe (Linear)
    # Zwraca 2 najlepszych z 3: C1 (avg), C2 (1.5*P1-0.5*P2), C3 (-0.5*P1+1.5*P2)
    child_lin1, child_lin2 = RealCrossover.linear(
        parent_real1,
        parent_real2,
        crossover_probability=P_c,
        fitness_function=polynomial_3rd_degree_fitness,
        minimize=True
    )
    # Wartości C1, C2, C3 przed wyborem:
    # C1: [0.0, 1.0]. Fitness: 0^3 + 1^2 - 3*0 + 4 = 5.0
    # C2: [2.0, -0.5]. Fitness: 2^3 + (-0.5)^2 - 3*2 + 4 = 8 + 0.25 - 6 + 4 = 6.25
    # C3: [-2.0, 1.5]. Fitness: (-2)^3 + 1.5^2 - 3*(-2) + 4 = -8 + 2.25 + 6 + 4 = 4.25 (Najlepszy)
    print("=== 5.2 Krzyżowanie Liniowe ===")
    print(f"Potomek Linear 1 (Najlepszy): {child_lin1.genes} (Oczekiwane [-2.0, 1.5] lub [0.0, 1.0])")
    print(f"Fitness C1: {polynomial_3rd_degree_fitness(child_lin1.genes):.4f}")
    print(f"Potomek Linear 2: {child_lin2.genes}")
    print("-" * 40)

    # 5.3 Krzyżowanie alpha (Blend Alpha)
    # Używamy alpha=0.1, Pc=1.0
    child_bla1, child_bla2 = RealCrossover.blend_alpha(parent_real1, parent_real2, crossover_probability=P_c, alpha=0.1)
    print("=== 5.3 Krzyżowanie mieszające alpha (alpha=0.1) ===")
    print(f"Potomek alpha 1: {child_bla1.genes}")
    print(f"Potomek alpha 2: {child_bla2.genes}")
    # Weryfikacja zakresu x1: min=-1.0, max=1.0, range=2.0. α*range=0.2. Zakres losowania: [-1.2, 1.2]
    # Weryfikacja zakresu x2: min=0.5, max=1.5, range=1.0. α*range=0.1. Zakres losowania: [0.4, 1.6]
    print("-" * 40)

    # 5.4 Krzyżowanie alpha-beta (Blend Alpha Beta)
    # Używamy alpha=0.1, beta=0.5, Pc=1.0 (Asymetryczne rozszerzenie)
    child_blab1, child_blab2 = RealCrossover.blend_alpha_beta(parent_real1, parent_real2, crossover_probability=P_c,
                                                              alpha=0.1, beta=0.5)
    print("=== 5.4 Krzyżowanie mieszające alpha-beta (alpha=0.1, beta=0.5) ===")
    print(f"Potomek alpha-beta 1: {child_blab1.genes}")
    print(f"Potomek alpha-beta 2: {child_blab2.genes}")
    # Weryfikacja zakresu x1: min=-1.0, max=1.0, range=2.0. Lower: -1.0 - 0.2 = -1.2. Upper: 1.0 + 1.0 = 2.0. Zakres losowania: [-1.2, 2.0]
    print("-" * 40)

    # 5.5 Krzyżowanie Uśredniające (Averaging)
    # Child1 = Child2 = (P1 + P2) / 2
    child_avg1, child_avg2 = RealCrossover.averaging(parent_real1, parent_real2, crossover_probability=P_c)
    print("=== 5.5 Krzyżowanie Uśredniające ===")
    print(f"Potomek Averaging 1: {child_avg1.genes} (Powinno być [0.0, 1.0])")
    print(f"Potomek Averaging 2: {child_avg2.genes} (Powinno być [0.0, 1.0])")

    P_bounds_multi = [(0.0, 15.0), (0.0, 15.0)]
    alphas_multi = [0.2, 0.3, 0.5]

    # Tworzenie i ustawianie rodziców
    parent_x = RealChromosome(n_variables=2, bounds=P_bounds_multi)
    parent_y = RealChromosome(n_variables=2, bounds=P_bounds_multi)
    parent_z = RealChromosome(n_variables=2, bounds=P_bounds_multi)

    parent_x.genes = np.array([2.0, 3.0])  # Px
    parent_y.genes = np.array([4.0, 8.0])  # Py
    parent_z.genes = np.array([7.0, 9.0])  # Pz

    parents_list = [parent_x, parent_y, parent_z]

    print("\n\n=== 5.6 Krzyżowanie arytmetyczne wieloosobnicze ===")
    print(f"Rodzice (Px, Py, Pz): {parent_x.genes}, {parent_y.genes}, {parent_z.genes}")
    print(f"Wagi (Alphy): {alphas_multi}")

    new_child = RealCrossover.multi_parent_arithmetic(parents=parents_list, alphas=alphas_multi)

    print(f"\nNowy Potomek: {new_child.genes}")
    print(f"Oczekiwane Geny: [5.1, 7.5]")

    new_child_2 = RealCrossover.multi_parent_arithmetic(parents=parents_list, alphas=None)

    print(f"\nNowy Potomek: {new_child_2.genes}")

    expected_genes = np.array([5.1, 7.5])
    is_correct = np.allclose(new_child.genes, expected_genes)
    print(f"Weryfikacja: {'Zgodne' if is_correct else 'Niezgodne'}")


    print("\n--- Testy Real Crossover zakończone. ---")

    print("\n✓ Wszystkie testy zakończone")