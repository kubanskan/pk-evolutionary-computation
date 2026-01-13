import numpy as np
import matplotlib.pyplot as plt
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy import FloatVar
import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F52014

# 1. Funkcje celu

def sphere(solution):
    return np.sum(solution ** 2)

def schwefel(solution):
    return bf.Schwefel(n_dimensions=len(solution))(solution)

def cec2014_f5(solution):
    f = F52014(ndim=len(solution))
    return f.evaluate(solution)


# 2. Wybór funkcji przez użytkownika
functions = {
    "1": {"name": "sphere", "func": sphere, "bounds": (-500, 500)},
    "2": {"name": "schwefel", "func": schwefel, "bounds": (-500, 500)},
    "3": {"name": "cec2014_f5", "func": cec2014_f5, "bounds": (-100, 100)},
}


print("\nDostępne funkcje celu:")
for k, v in functions.items():
    print(f"{k} - {v['name']}")

choice = input("\nWybierz funkcję (numer lub nazwa): ").strip().lower()

selected = None

# wybór po numerze
if choice in functions:
    selected = functions[choice]

# wybór po nazwie
else:
    for v in functions.values():
        if choice == v["name"]:
            selected = v
            break

# fallback
if selected is None:
    print("Niepoprawny wybór, używam domyślnie: sphere")
    selected = functions["1"]

objective_function = selected["func"]
BOUNDS = selected["bounds"]

print(f"Wybrano funkcję: {selected['name']}")

# 3. Definicja problemu
N_DIMENSIONS = 10
if choice == "cec2014_f5":
    BOUNDS = (-100, 100)
else:
    BOUNDS = (-500, 500)


bounds = FloatVar(
    lb=[BOUNDS[0]] * N_DIMENSIONS,
    ub=[BOUNDS[1]] * N_DIMENSIONS
)

problem = {
    "obj_func": objective_function,
    "bounds": bounds,
    "minmax": "min"
}


# 4. Parametry algorytmu WOA
EPOCHS = 100
POP_SIZE = 80
N_RUNS = 10  # Do wykresu średniej ± std


# 5. Wykonanie wielu uruchomień dla wykresów
all_best = []

for run in range(N_RUNS):
    model = OriginalWOA(epoch=EPOCHS, pop_size=POP_SIZE, verbose=False)
    model.solve(problem)
    all_best.append(model.history.list_global_best_fit)

all_best = np.array(all_best)
avg = np.mean(all_best, axis=0)
std = np.std(all_best, axis=0)
generations = np.arange(1, EPOCHS + 1)


# 6. Rysowanie wykresów
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(generations, all_best[-1], 'b-', linewidth=2, label='Najlepsza z ostatniego run')
axes[0].set_xlabel('Generacja')
axes[0].set_ylabel('Wartość funkcji')
axes[0].set_title('Najlepsza wartość funkcji celu od kolejnej iteracji')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(generations, avg, 'g-', linewidth=2, label='Średnia')
axes[1].fill_between(
    generations,
    avg - std,
    avg + std,
    alpha=0.3,
    color='green',
    label='±1 std'
)
axes[1].set_xlabel('Generacja')
axes[1].set_ylabel('Wartość funkcji')
axes[1].set_title('Średnia wartość funkcji i odchylenie standardowe')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()