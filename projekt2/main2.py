import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from projekt2.app.database import DataBase
from projekt2.app.ui import BENCHMARK_FUNCTIONS
from projekt2.ga.genecticalgorithm import GeneticAlgorithmConfig
from projekt2.app.config import GAConfig


def run_single_set(params, repetitions, output_dir, set_index):
    Path(output_dir).mkdir(exist_ok=True)
    db_manager = DataBase()
    func_name = params['function']
    func_info = BENCHMARK_FUNCTIONS[func_name]
    func = func_info["function"]

    # dynamiczna aktualizacja wymiaru funkcji
    if hasattr(func, "update_dimension"):
        func.update_dimension(params['num_variables'])
    elif callable(func):
        func(np.zeros(params['num_variables']))

    func_info["bounds"] = func.bounds
    func_info["optimum"] = func.optimum

    print(f"Zestaw {set_index + 1}: {params}", flush=True)
    results_for_set = []

    objective_wrapper = lambda x: func(x)

    for r in range(repetitions):
        print(f" Uruchomienie {r + 1}/{repetitions}", flush=True)
        config = GAConfig(
            population_size=params.get('population_size', 100),
            num_generations=params.get('num_generations', 100),
            num_variables=params.get('num_variables', 10),
            precision=params.get('precision', 16),
            crossover_prob=params.get('crossover_prob', 0.8),
            mutation_prob=params.get('mutation_prob', 0.01),
            inversion_prob=params.get('inversion_prob', 0.05),
            elite_size=params.get('elite_size', 2),
            selection_method=params.get('selection_method', 'tournament'),
            tournament_size=params.get('tournament_size', 3),
            crossover_method=params.get('crossover_method', 'one_point'),
            mutation_method=params.get('mutation_method', 'one_point'),
            representation=params.get('representation', "binary"),
            arithmetic_alpha=params.get('arithmetic_alpha'),
            blend_alpha=params.get('blend_alpha'),
            blend_alpha_param=params.get('blend_alpha_param'),
            blend_beta_param=params.get('blend_beta_param'),
            mutation_range=params.get('mutation_range'),
            gaussian_sigma=params.get('gaussian_sigma'),
            num_parents=params.get('num_parents', 2),
            selection_percentage=params.get('selection_percentage'),
            bounds=func_info["bounds"],
            optimization_type=params.get('optimization_type', 'minimize')
        )

        ga = GeneticAlgorithmConfig(config, objective_wrapper)
        result = ga.evolve()
        db_manager.save_run(func_name, config.num_variables, result)

        results_for_set.append({
            "run_index": r + 1,
            "params": {
                "population_size": config.population_size,
                "num_generations": config.num_generations,
                "num_variables": config.num_variables,
                "precision": config.precision,
                "elite_size": config.elite_size,
                "selection_method": config.selection_method,
                "tournament_size": config.tournament_size,
                "selection_pct": params.get("selection_pct"),
                "crossover_method": config.crossover_method,
                "crossover_prob": config.crossover_prob,
                "mutation_method": config.mutation_method,
                "mutation_prob": config.mutation_prob,
                "inversion_prob": config.inversion_prob
            },
            "result": result
        })

    # analiza wyników
    best_run = min(results_for_set, key=lambda x: x['result']['best_fitness'])
    worst_run = max(results_for_set, key=lambda x: x['result']['best_fitness'])
    avg_best = np.mean([res['result']['best_fitness'] for res in results_for_set])
    avg_time = np.mean([res['result']['elapsed_time'] for res in results_for_set])

    # zapis wyników do pliku JSON
    output_file = Path(output_dir) / f"{func_name}_{params['num_variables']}_params_set{set_index}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "params": params,
            "results": results_for_set,
            "best_run": best_run,
            "worst_run": worst_run,
            "avg_best_fitness": avg_best,
            "avg_time": avg_time,
            "optimum": func_info['optimum']
        }, f, indent=4)

    # wykresy dla najlepszego uruchomienia
    history = best_run['result']['history']
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(history['generation'], history['best_fitness'], 'b-', linewidth=2)
    axes[0].set_xlabel('Generacja')
    axes[0].set_ylabel('Wartość funkcji')
    axes[0].set_title(f"{func_name} - najlepsze uruchomienie")
    axes[0].grid(True)
    generations = np.array(history['generation'])
    avg = np.array(history['avg_fitness'])
    std = np.array(history['std_fitness'])
    axes[1].plot(generations, avg, 'g-', linewidth=2)
    axes[1].fill_between(generations, avg - std, avg + std, alpha=0.3, color='green')
    axes[1].set_xlabel('Generacja')
    axes[1].set_ylabel('Średnia ± std')
    axes[1].set_title('Średnia wartość funkcji i odchylenie standardowe')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{func_name}_{params['num_variables']}_params_set{set_index}_plots.png")
    plt.close()

    return {
        "function": func_name,
        "num_variables": params['num_variables'],
        **params,
        "best_run": best_run['result']['best_fitness'],
        "worst_run": worst_run['result']['best_fitness'],
        "avg_best": avg_best,
        "avg_time": avg_time,
        "optimum": func_info['optimum'],
        "error": abs(best_run['result']['best_fitness'] - func_info['optimum'])
    }

def batch_run_parallel(param_sets, repetitions=10, output_dir="batch_results"):
    Path(output_dir).mkdir(exist_ok=True)
    final_summary = []
    MAX_WORKERS = 4
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single_set, params, repetitions, output_dir, i)
                   for i, params in enumerate(param_sets)]
        for future in as_completed(futures):
            final_summary.append(future.result())

    report_file = Path(output_dir) / "summary_report.json"
    with open(report_file, 'w') as f:
        json.dump(final_summary, f, indent=4)

    print("Batch run zakończony. Wszystkie wyniki zapisane.", flush=True)


param_sets = [
    {
        "function": "Schwefel",
        "representation": "real",
        "num_variables": 2,
        "population_size": 1000,
        "num_generations": 200,
        "elite_size": 3,
        "selection_method": "tournament",
        "tournament_size": 3,
        "crossover_method": "arithmetic",
        "crossover_prob": 0.85,
        "arithmetic_alpha": 0.5,
        "mutation_method": "uniform",
        "mutation_prob": 0.05,
        "mutation_range": 0.1,
        "optimization_type": "minimize"
    },
    {
        "function": "Schwefel",
        "representation": "real",
        "num_variables": 5,
        "population_size": 2000,
        "num_generations": 200,
        "elite_size": 3,
        "selection_method": "best",
        "selection_percentage": 0.5,
        "crossover_method": "blend_alpha",
        "crossover_prob": 0.9,
        "blend_alpha": 0.5,
        "mutation_method": "gaussian",
        "mutation_prob": 0.3,
        "gaussian_sigma": 0.1,
        "optimization_type": "minimize"
    },
    {
        "function": "Schwefel",
        "representation": "real",
        "num_variables": 10,
        "population_size": 3000,
        "num_generations": 200,
        "elite_size": 5,
        "selection_method": "roulette",
        "crossover_method": "linear",
        "crossover_prob": 0.85,
        "mutation_method": "uniform",
        "mutation_prob": 0.2,
        "mutation_range": 0.15,
        "optimization_type": "minimize"
    },
    {
        "function": "Schwefel",
        "representation": "real",
        "num_variables": 2,
        "population_size": 1000,
        "num_generations": 200,
        "elite_size": 5,
        "selection_method": "tournament",
        "tournament_size": 5,
        "crossover_method": "averaging",
        "crossover_prob": 0.85,
        "mutation_method": "uniform",
        "mutation_prob": 0.15,
        "mutation_range": 0.15,
        "optimization_type": "minimize"
    },
    {
        "function": "Schwefel",
        "representation": "real",
        "num_variables": 10,
        "population_size": 3000,
        "num_generations": 200,
        "elite_size": 3,
        "selection_method": "tournament",
        "tournament_size": 8,
        "crossover_method": "blend_alpha",
        "crossover_prob": 0.9,
        "blend_alpha": 0.5,
        "mutation_method": "gaussian",
        "mutation_prob": 0.4,
        "gaussian_sigma": 0.1,
        "optimization_type": "minimize"
    },
    {
        "function": "Schwefel",
        "representation": "real",
        "num_variables": 10,
        "population_size": 500,
        "num_generations": 200,
        "elite_size": 2,
        "selection_method": "tournament",
        "tournament_size": 3,
        "crossover_method": "arithmetic",
        "crossover_prob": 0.85,
        "arithmetic_alpha": 0.5,
        "mutation_method": "uniform",
        "mutation_prob": 0.1,
        "mutation_range": 0.15,
        "optimization_type": "minimize"
    },
    {
        "function": "Schwefel",
        "representation": "real",
        "num_variables": 2,
        "population_size": 1000,
        "num_generations": 300,
        "elite_size": 3,
        "selection_method": "tournament",
        "tournament_size": 4,
        "crossover_method": "blend_alpha",
        "crossover_prob": 0.8,
        "blend_alpha": 0.5,
        "mutation_method": "uniform",
        "mutation_prob": 0.15,
        "mutation_range": 0.2,
        "optimization_type": "minimize"
    }
]

if __name__ == "__main__":
    batch_run_parallel(param_sets, repetitions=10, output_dir="batch_results_20")