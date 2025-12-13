#na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging
import pygad
import numpy
import benchmark_functions as bf
from decoder import decode_with_binary_chromosome
from chromosome import BinaryChromosome
from opfunu.cec_based.cec2014 import F52014


#Konfiguracja algorytmu genetycznego

func = bf.Schwefel(n_dimensions=2)

N_VARIABLES = 2
BOUNDS = [(-500, 500), (-500, 500)]
PRECISION = 6

tmp_chrom = BinaryChromosome(
    n_variables=N_VARIABLES,
    bounds=BOUNDS,
    precision=PRECISION
)
num_genes = tmp_chrom.total_length

print(func.suggested_bounds())
print(func.minimum())

def fitness_func(ga_instance, solution, solution_idx):
    x = decode_with_binary_chromosome(
        solution,
        n_variables=N_VARIABLES,
        bounds=BOUNDS,
        precision=PRECISION
    )
    value = func(x)
    return 1.0 / (value + 1e-8)

fitness_function = fitness_func
num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
init_range_low = 0
init_range_high = 2
mutation_num_genes = 1
parent_selection_type = "tournament"
crossover_type = "two_points" #"uniform"
mutation_type = "swap" #"random"


#Konfiguracja logowania

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

def on_generation(ga_instance):
    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    ga_instance.logger.info("Best    = {fitness}".format(fitness=1./solution_fitness))
    ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))

    tmp = [1./x for x in ga_instance.last_generation_fitness] #ponownie odwrotność by zrobić sobie dobre statystyki

    ga_instance.logger.info("Min    = {min}".format(min=numpy.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=numpy.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=numpy.average(tmp)))
    ga_instance.logger.info("Std    = {std}".format(std=numpy.std(tmp)))
    ga_instance.logger.info("\r\n")


#Właściwy algorytm genetyczny

ga_instance = pygad.GA(num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    num_genes=num_genes,
                    fitness_func=fitness_func,
                    init_range_low=0,
                    init_range_high=2,
                    gene_type=int,
                    mutation_num_genes=mutation_num_genes,
                    parent_selection_type=parent_selection_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    keep_elitism=1,
                    K_tournament=3,
                    random_mutation_max_val=32.768,
                    random_mutation_min_val=-32.768,
                    logger=logger,
                    on_generation=on_generation,
                    parallel_processing=['thread', 4])

ga_instance.run()


best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1./solution_fitness))


# sztuczka: odwracamy by narysował nam się oczekiwany wykres dla problemu minimalizacji
ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()


