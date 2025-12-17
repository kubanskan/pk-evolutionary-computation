import numpy as np
import json
import time
import pygad
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F52014
from chromosome import BinaryChromosome
from decoder import decode_with_binary_chromosome
import real_operators as ro
matplotlib.use('Agg')


def get_problem_settings(func_name, n_vars):
    """
    Zwraca obiekt funkcji, granice oraz (opcjonalnie) optimum.
    """
    optimum = None
    if func_name == "schwefel":
        func_obj = bf.Schwefel(n_dimensions=n_vars)
        bounds = [(-500, 500)] * n_vars
        optimum = 0.0
    elif "cec2014" in func_name.lower():
        func_obj = F52014(ndim=n_vars)
        bounds = [(-100, 100)] * n_vars
        optimum = getattr(func_obj, "f_global", None)
    else:
        raise ValueError(f"Nieznana funkcja: {func_name}")

    return func_obj, bounds, optimum


def run_single_experiment(params, run_id, output_dir, set_index):
    """
    Uruchamia pojedynczą instancję PyGAD.
    Zwraca pełny słownik wyników zgodny ze starym formatem.
    """
    start_time = time.time()

    func_name = params['function']
    n_vars = params['num_variables']

    func_obj, bounds, _ = get_problem_settings(func_name, n_vars)

    def fitness_func(ga_inst, solution, solution_idx):

        if params['representation'] == 'binary':
            decoded = decode_with_binary_chromosome(solution, n_vars, bounds, params.get('precision', 6))
        else:
            decoded = solution


        try:
            val = func_obj(decoded)
        except:
            val = func_obj.evaluate(decoded)


        return 1.0 / (np.abs(val) + 1e-8)


    history = {
        'generation': [],
        'best_fitness': [],
        'avg_fitness': [],
        'std_fitness': []
    }

    def on_generation(ga_inst):
        gen = ga_inst.generations_completed
        fitness_values = ga_inst.last_generation_fitness
        real_scores = [(1.0 / f) - 1e-8 for f in fitness_values]

        best_val = min(real_scores)
        avg_val = np.mean(real_scores)
        std_val = np.std(real_scores)

        history['generation'].append(int(gen))
        history['best_fitness'].append(float(best_val))
        history['avg_fitness'].append(float(avg_val))
        history['std_fitness'].append(float(std_val))

    crossover_type = params.get('crossover_method', 'two_points')
    mutation_type = params.get('mutation_method', 'swap')

    if params['representation'] == 'binary':
        precision = params.get('precision', 6)
        tmp_chrom = BinaryChromosome(n_vars, bounds, precision)
        num_genes = tmp_chrom.total_length
        gene_type = int
        gene_space = [0, 1]
        init_range_low, init_range_high = 0, 2

    else:
        num_genes = n_vars
        gene_type = float
        gene_space = None
        init_range_low, init_range_high = bounds[0]

        cross_map = {
            "arithmetic": ro.crossover_arithmetic,
            "blend_alpha": ro.crossover_blend_alpha,
            "blend_beta": ro.crossover_blend_alpha_beta,
            "linear": ro.crossover_linear,
            "average": ro.crossover_averaging,
            "split": ro.crossover_simple_split
        }
        mut_map = {
            "gaussian": ro.mutation_gaussian,
            "uniform": ro.mutation_uniform
        }

        if params['crossover_method'] in cross_map:
            crossover_type = cross_map[params['crossover_method']]
        if params['mutation_method'] in mut_map:
            mutation_type = mut_map[params['mutation_method']]

    ga_instance = pygad.GA(
        num_generations=params['num_generations'],
        num_parents_mating=int(params['population_size'] * params.get('parents_ratio', 0.5)),
        fitness_func=fitness_func,
        sol_per_pop=params['population_size'],
        num_genes=num_genes,
        gene_type=gene_type,
        gene_space=gene_space,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=params.get('selection_method', 'tournament'),
        K_tournament=params.get('tournament_size', 3),
        crossover_type=crossover_type,
        crossover_probability=params.get('crossover_prob', 0.8),
        mutation_type=mutation_type,
        mutation_probability=params.get('mutation_prob', 0.1),
        on_generation=on_generation,
        keep_elitism=params.get('elite_size', 1),
        suppress_warnings=True
    )

    ga_instance.custom_alpha = params.get('arithmetic_alpha', 0.5)
    ga_instance.custom_beta = params.get('blend_beta_param', 0.5)
    ga_instance.custom_sigma = params.get('gaussian_sigma', 1.0)
    ga_instance.custom_mut_range = params.get('mutation_range', 0.1)
    ga_instance.custom_cross_prob = params.get('crossover_prob', 0.8)
    ga_instance.custom_mut_prob = params.get('mutation_prob', 0.1)

    ga_instance.run()


    solution, solution_fitness, _ = ga_instance.best_solution()

    if params['representation'] == 'binary':
        decoded_sol = decode_with_binary_chromosome(solution, n_vars, bounds, params.get('precision', 6)).tolist()
    else:
        decoded_sol = solution.tolist() if isinstance(solution, np.ndarray) else solution

    best_val_func = (1.0 / solution_fitness) - 1e-8
    elapsed_time = time.time() - start_time

    run_result_entry = {
        "run_index": run_id,
        "params": params,
        "result": {
            "best_fitness": float(best_val_func),
            "best_solution": decoded_sol,
            "elapsed_time": elapsed_time,
            "history": history
        }
    }

    return run_result_entry


def process_set(params, repetitions, output_dir, set_index):
    """
    Przetwarza zestaw, zbiera wyniki i zapisuje JSON w pełnym formacie.
    """
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Start Zestawu {set_index}: {params['function']} ({params['representation']})", flush=True)

    _, _, optimum = get_problem_settings(params['function'], params['num_variables'])

    results_for_set = []


    for i in range(repetitions):
        res = run_single_experiment(params, i + 1, output_dir, set_index)
        results_for_set.append(res)


    best_run = min(results_for_set, key=lambda x: x['result']['best_fitness'])
    worst_run = max(results_for_set, key=lambda x: x['result']['best_fitness'])

    all_best_vals = [r['result']['best_fitness'] for r in results_for_set]
    all_times = [r['result']['elapsed_time'] for r in results_for_set]

    avg_best = np.mean(all_best_vals)
    avg_time = np.mean(all_times)

    final_summary = {
        "params": params,
        "results": results_for_set,
        "best_run": best_run,
        "worst_run": worst_run,
        "avg_best_fitness": float(avg_best),
        "avg_time": float(avg_time),
        "optimum": optimum if optimum is not None else "Unknown"
    }

    filename_base = f"set{set_index}_{params['function']}_{params['representation']}"
    with open(Path(output_dir) / f"{filename_base}_results.json", 'w') as f:
        json.dump(final_summary, f, indent=4)

    hist = best_run['result']['history']
    gens = np.array(hist['generation'])
    best_fit = np.array(hist['best_fitness'])
    avg_fit = np.array(hist['avg_fitness'])
    std_fit = np.array(hist['std_fitness'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(gens, best_fit, 'b-', linewidth=2, label='Najlepszy wynik')
    axes[0].set_xlabel('Generacja')
    axes[0].set_ylabel('Wartość funkcji celu (min)')
    axes[0].set_title(f'Zestaw {set_index} - Najlepsze uruchomienie\nWynik: {best_run["result"]["best_fitness"]:.5f}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(gens, avg_fit, 'g-', linewidth=2, label='Średnia populacji')
    axes[1].fill_between(gens, avg_fit - std_fit, avg_fit + std_fit, alpha=0.3, color='green', label='±1 STD')
    axes[1].set_xlabel('Generacja')
    axes[1].set_ylabel('Wartość funkcji')
    axes[1].set_title('Różnorodność populacji (Średnia ± STD)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{filename_base}_plot.png")
    plt.close()

    print(f"-> Zakończono Zestaw {set_index}. Średnia: {avg_best:.4f}", flush=True)
    return final_summary


def batch_run_parallel(param_sets, repetitions=5, output_dir="results_batch"):
    """
    Główna pętla sterująca.
    """
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Rozpoczynanie batch run. Zestawów: {len(param_sets)}, Powtórzeń: {repetitions}")

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, p in enumerate(param_sets):
            futures.append(executor.submit(process_set, p, repetitions, output_dir, i + 1))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"!!! Błąd w procesie: {e}")

    print("\n=== WSZYSTKIE OBLICZENIA ZAKOŃCZONE ===")


param_sets = [
    {
        "function": "schwefel",
        "representation": "binary",
        "num_variables": 10,
        "precision": 6,
        "population_size": 50,
        "num_generations": 100,
        "crossover_method": "two_points",
        "crossover_prob": 0.8,
        "mutation_method": "swap",
        "mutation_prob": 0.05,
        "selection_method": "tournament",
        "tournament_size": 3,
        "elite_size": 2
    },
    {
        "function": "schwefel",
        "representation": "real",
        "num_variables": 10,
        "population_size": 50,
        "num_generations": 100,
        "crossover_method": "arithmetic",
        "arithmetic_alpha": 0.5,
        "crossover_prob": 0.9,
        "mutation_method": "gaussian",
        "gaussian_sigma": 1.0,
        "mutation_prob": 0.1,
        "selection_method": "tournament",
        "tournament_size": 3,
        "elite_size": 2
    },
    {
        "function": "cec2014",
        "representation": "real",
        "num_variables": 10,
        "population_size": 100,
        "num_generations": 200,
        "crossover_method": "blend_alpha",
        "arithmetic_alpha": 0.5,
        "crossover_prob": 0.8,
        "mutation_method": "uniform",
        "mutation_range": 0.1,
        "mutation_prob": 0.2,
        "selection_method": "rws",
        "elite_size": 2
    },
    {
        "function": "cec2014",
        "representation": "real",
        "num_variables": 10,
        "population_size": 100,
        "num_generations": 150,
        "crossover_method": "linear",
        "crossover_prob": 0.9,
        "mutation_method": "gaussian",
        "gaussian_sigma": 0.5,
        "mutation_prob": 0.1,
        "selection_method": "tournament",
        "tournament_size": 5,
        "elite_size": 2
    }
]

if __name__ == "__main__":
    batch_run_parallel(param_sets, repetitions=10, output_dir="moje_wyniki_batch")