import numpy as np
import matplotlib.pyplot as plt
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy import FloatVar
import benchmark_functions as bf

# 1. Funkcje celu

def sphere(solution):
    return np.sum(solution ** 2)

def schwefel(solution):
    return bf.Schwefel(n_dimensions=len(solution))(solution)

# 2. Wybór funkcji przez użytkownika
func_dict = {
    "sphere": sphere,
    "schwefel": schwefel
}

choice = input("Wybierz funkcję celu (sphere/schwefel): ").strip().lower()

if choice not in func_dict:
    print("Niepoprawny wybór, używam domyślnie 'sphere'")
    choice = "sphere"

objective_function = func_dict[choice]
print(f"Wybrano funkcję: {choice}")

# 3. Definicja problemu
N_DIMENSIONS = 2
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