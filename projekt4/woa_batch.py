import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from mealpy.swarm_based.WOA import OriginalWOA
from mealpy import FloatVar
import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F52014


# Funkcje celu ===========================================

def sphere(solution):
    return np.sum(solution ** 2)


def schwefel(solution):
    return bf.Schwefel(n_dimensions=len(solution))(solution)

"""
def schwefel(solution):
    #Schwefel przesunięty, minimum w 0
    n = len(solution)
    raw = bf.Schwefel(n_dimensions=n)(solution)
    # przesunięcie aby minimum było 0
    return raw - 418.9829 * n
"""

class CEC2014F5Wrapper:
    def __init__(self, ndim):
        self.func = F52014(ndim=ndim)

    def __call__(self, solution):
        return self.func.evaluate(solution)


# Funkcje benchmarkowe + bounds + optimum ============================================

def schwefel_optimum(ndim):
    func = bf.Schwefel(n_dimensions=ndim)
    if hasattr(func, 'minimum'):
        res = func.minimum()
        # Sprawdzamy, czy to specyficzny obiekt 'Optimum' lub krotka
        # Próbujemy wyciągnąć pierwszy element (wartość funkcji)
        try:
            return float(res[0])
        except (TypeError, IndexError):
            return float(res)
    return 0.0


def cec2014_f5_optimum(ndim):
    func = F52014(ndim=ndim)
    return func.f_global


FUNCTIONS = {
    "sphere": {
        "func_factory": lambda ndim: sphere,
        "bounds": (-500, 500),
        "optimum_factory": lambda ndim: 0.0
    },
    "schwefel": {
        "func_factory": lambda ndim: schwefel,
        "bounds": (-500, 500),
        "optimum_factory": schwefel_optimum
    },
    "cec2014_f5": {
        "func_factory": lambda ndim: CEC2014F5Wrapper(ndim),
        "bounds": (-100, 100),
        "optimum_factory": cec2014_f5_optimum
    }
}


# Pojedynczy zestaw parametrów ==================================

def run_single_set_woa(params, repetitions, output_dir, set_index):
    Path(output_dir).mkdir(exist_ok=True)

    func_name = params["function"]
    n_dim = params["num_variables"]
    epochs = params["epochs"]
    pop_size = params["pop_size"]

    func_info = FUNCTIONS[func_name]
    objective_function = func_info["func_factory"](n_dim)
    bounds_range = func_info["bounds"]
    optimum = func_info["optimum_factory"](n_dim)

    bounds = FloatVar(
        lb=[bounds_range[0]] * n_dim,
        ub=[bounds_range[1]] * n_dim
    )

    problem = {
        "obj_func": objective_function,
        "bounds": bounds,
        "minmax": "min"
    }

    print(f"Zestaw {set_index + 1}: {params}", flush=True)

    results_for_set = []

    for r in range(repetitions):
        print(f"  Uruchomienie {r + 1}/{repetitions}", flush=True)

        model = OriginalWOA(epoch=epochs, pop_size=pop_size, verbose=False)
        model.solve(problem)

        history = np.array(model.history.list_global_best_fit)
        elapsed_time = float(np.sum(model.history.list_epoch_time))

        results_for_set.append({
            "run_index": r + 1,
            "best_fitness": history[-1],
            "history": history.tolist(),
            "elapsed_time": elapsed_time
        })

    all_histories = np.array([r["history"] for r in results_for_set])
    best_run = min(results_for_set, key=lambda x: x["best_fitness"])
    worst_run = max(results_for_set, key=lambda x: x["best_fitness"])
    avg_best = float(np.mean([r["best_fitness"] for r in results_for_set]))
    avg_time = float(np.mean([r["elapsed_time"] for r in results_for_set]))

    history_stats = {
        "generation": list(range(1, epochs + 1)),
        "min": np.min(all_histories, axis=0).tolist(),
        "max": np.max(all_histories, axis=0).tolist(),
        "mean": np.mean(all_histories, axis=0).tolist(),
        "std": np.std(all_histories, axis=0).tolist()
    }

    # Zapis JSON ============================
    output_file = Path(output_dir) / f"WOA_{func_name}_{n_dim}D_set{set_index}_results.json"

    with open(output_file, "w") as f:
        json.dump({
            "params": params,
            "bounds": bounds_range,
            "optimum": optimum,
            "runs": results_for_set,
            "history_statistics": history_stats,
            "best_run": best_run,
            "worst_run": worst_run,
            "avg_best": avg_best,
            "avg_time": avg_time,
            "error": abs(best_run["best_fitness"] - optimum)
        }, f, indent=4)

    # Wykres zbiorczy ============================
    gen = np.array(history_stats["generation"])
    mean = np.array(history_stats["mean"])
    std = np.array(history_stats["std"])
    min_v = np.array(history_stats["min"])
    max_v = np.array(history_stats["max"])

    plt.figure(figsize=(10, 6))
    plt.plot(gen, mean, label="Średnia best fitness", linewidth=2)
    plt.fill_between(gen, mean - std, mean + std, alpha=0.3, label="±1 std")
    plt.plot(gen, min_v, "--", linewidth=1, label="Min")
    plt.plot(gen, max_v, "--", linewidth=1, label="Max")

    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji")
    plt.title(f"WOA – {func_name}, {n_dim} zmiennych (epoki: {epochs}, pop: {pop_size})\nNajlepszy wynik: {best_run['best_fitness']:.6e}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(Path(output_dir) / f"WOA_{func_name}_{n_dim}_set{set_index}_convergence.png")
    plt.close()

    return {
        "function": func_name,
        "num_variables": n_dim,
        "best": best_run["best_fitness"],
        "worst": worst_run["best_fitness"],
        "avg_best": avg_best,
        "avg_time": avg_time,
        "optimum": optimum,
        "error": abs(best_run["best_fitness"] - optimum)
    }


def batch_run_parallel_woa(param_sets, repetitions=10, output_dir="woa_results"):
    Path(output_dir).mkdir(exist_ok=True)
    summary = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_single_set_woa, params, repetitions, output_dir, i)
            for i, params in enumerate(param_sets)
        ]

        for future in as_completed(futures):
            summary.append(future.result())

    with open(Path(output_dir) / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Batch WOA zakończony. Wszystkie wyniki zapisane.", flush=True)


param_sets = [
    # Schwefel
    {"function": "schwefel", "num_variables": 2, "epochs": 50, "pop_size": 100},
    {"function": "schwefel", "num_variables": 2, "epochs": 50, "pop_size": 500},
    {"function": "schwefel", "num_variables": 10, "epochs": 300, "pop_size": 500},
    {"function": "schwefel", "num_variables": 10, "epochs": 500, "pop_size": 800},
    # CEC2014 F5
    {"function": "cec2014_f5", "num_variables": 10, "epochs": 100, "pop_size": 400},
    {"function": "cec2014_f5", "num_variables": 10, "epochs": 200, "pop_size": 800},
    {"function": "cec2014_f5", "num_variables": 20, "epochs": 300, "pop_size": 1500},
    {"function": "cec2014_f5", "num_variables": 10, "epochs": 500, "pop_size": 1000},
]



if __name__ == "__main__":
    batch_run_parallel_woa(param_sets, repetitions=10, output_dir="woa_batch_results_2")