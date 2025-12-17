import sys
import logging
import pygad
import numpy as np
import matplotlib.pyplot as plt
import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F52014
from decoder import decode_with_binary_chromosome
from chromosome import BinaryChromosome
from real_operators import (
    crossover_arithmetic,
    crossover_blend_alpha,
    crossover_blend_alpha_beta,
    crossover_averaging,
    crossover_linear,
    crossover_simple_split,
    mutation_gaussian,
    mutation_uniform
)


def plot_results(history, title_suffix=""):
    """Generuje dwa wykresy: postęp najlepszego wyniku oraz statystyki populacji"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))


    axes[0].plot(history['generation'], history['best_fitness'], 'b-', linewidth=2, label='Najlepsza')
    axes[0].set_xlabel('Generacja')
    axes[0].set_ylabel('Wartość funkcji celu')
    axes[0].set_title(f'Postęp optymalizacji: {title_suffix}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()


    gens = np.array(history['generation'])
    avg = np.array(history['avg_fitness'])
    std = np.array(history['std_fitness'])

    axes[1].plot(gens, avg, 'g-', linewidth=2, label='Średnia populacji')
    axes[1].fill_between(gens, avg - std, avg + std, alpha=0.3, color='green', label='±1 std (Rozrzut)')
    axes[1].set_xlabel('Generacja')
    axes[1].set_ylabel('Wartość funkcji')
    axes[1].set_title('Różnorodność populacji (Średnia ± STD)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# POMOCNICZE FUNKCJE WEJŚCIA
# ==========================================
def pobierz_parametr(tekst, domyslna, typ=int):
    odp = input(f"{tekst} [Domyślnie: {domyslna}]: ").strip()
    if not odp: return domyslna
    try:
        return typ(odp)
    except ValueError:
        return domyslna


def pobierz_wybor(tekst, opcje, domyslna):
    print(tekst)
    klucze = list(opcje.keys())
    for i, (klucz, opis) in enumerate(opcje.items(), 1):
        print(f"   {i}. {opis} ({klucz})")
    odp = input(f"Wybierz numer lub nazwę [Domyślnie: {domyslna}]: ").strip()
    if not odp: return domyslna
    if odp.isdigit():
        idx = int(odp) - 1
        if 0 <= idx < len(klucze): return klucze[idx]
    if odp in opcje: return odp
    return domyslna


# ==========================================
# KONFIGURACJA LOGOWANIA
# ==========================================
logger = logging.getLogger('ga_logger')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('Gen %(message)s'))
    logger.addHandler(ch)


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("   ZAAWANSOWANY KONFIGURATOR ALGORYTMU GENETYCZNEGO")
    print("=" * 60)


    history = {
        'generation': [],
        'best_fitness': [],
        'avg_fitness': [],
        'std_fitness': []
    }


    print("\n[1] KONFIGURACJA PROBLEMU")
    func_opts = {"schwefel": "Schwefel", "cec2014": "CEC2014 F5"}
    wybor_funkcji = pobierz_wybor("Funkcja celu:", func_opts, "schwefel")
    n_vars = pobierz_parametr("Liczba zmiennych (wymiar)", 2, int)

    if wybor_funkcji == "schwefel":
        func_obj = bf.Schwefel(n_dimensions=n_vars)
        single_bound = (-500, 500)
    else:
        func_obj = F52014(ndim=n_vars)
        single_bound = (-100, 100)
    bounds = [single_bound] * n_vars

    # --- 2. WYBÓR REPREZENTACJI ---
    print("\n[2] REPREZENTACJA OSOBNIKA")
    rep_opts = {"binarna": "Binarna (0/1)", "rzeczywista": "Rzeczywista (float)"}
    typ_rep = pobierz_wybor("Typ kodowania:", rep_opts, "binarna")

    precision = 6
    gene_space = None

    if typ_rep == "binarna":
        precision = pobierz_parametr("Dokładność (miejsca po przecinku)", 6, int)
        tmp = BinaryChromosome(n_vars, bounds, precision)
        num_genes = tmp.total_length
        gene_type = int
        gene_space = [0, 1]
        init_l, init_h = 0, 2
        print(f"-> Chromosom: {num_genes} bitów")
    else:
        num_genes = n_vars
        gene_type = float
        init_l, init_h = single_bound[0], single_bound[1]
        print(f"-> Chromosom: {num_genes} liczb rzeczywistych")

    # --- 3. PARAMETRY GA ---
    print("\n[3] PARAMETRY ALGORYTMU")
    adv = input("Czy chcesz ręcznie ustawić wszystkie parametry? (t/n) [n]: ").lower().strip()

    p_gen = 50;
    p_pop = 50;
    p_par = 25
    p_sel = "tournament";
    p_cross = "two_points";
    p_mut = "swap"
    u_k_tournament = 3;
    u_cross_prob = 0.8;
    u_mut_prob = 0.05
    u_alpha = 0.5;
    u_beta = 0.5;
    u_sigma = 1.0;
    u_mut_range = 0.1

    if adv == 't':
        p_gen = pobierz_parametr("Liczba pokoleń", 50, int)
        p_pop = pobierz_parametr("Rozmiar populacji", 50, int)
        p_par = pobierz_parametr("Liczba rodziców", int(p_pop / 2), int)
        sel_opts = {"tournament": "Turniejowa", "rws": "Koło Ruletki", "random": "Losowa"}
        p_sel = pobierz_wybor("Typ selekcji:", sel_opts, "tournament")
        if p_sel == "tournament":
            u_k_tournament = pobierz_parametr("Rozmiar turnieju (K)", 3, int)

        if typ_rep == "binarna":
            p_cross = pobierz_wybor("Metoda krzyżowania:",
                                    {"single_point": "1-pkt", "two_points": "2-pkt", "uniform": "Uniform"},
                                    "two_points")
            u_cross_prob = pobierz_parametr("Prawdopodobieństwo krzyżowania", 0.8, float)
            p_mut = pobierz_wybor("Metoda mutacji:", {"random": "Random (Flip)", "swap": "Swap"}, "swap")
            u_mut_prob = pobierz_parametr("Prawdopodobieństwo mutacji", 0.05, float)
        else:
            real_map = {"arithmetic": crossover_arithmetic, "blend": crossover_blend_alpha,
                        "blend_beta": crossover_blend_alpha_beta, "average": crossover_averaging,
                        "linear": crossover_linear, "split": crossover_simple_split}
            cross_key = pobierz_wybor("Metoda krzyżowania:", {"arithmetic": "Arytmetyczne", "blend": "BLX-Alpha",
                                                              "blend_beta": "BLX-Alpha-Beta", "average": "Uśredniające",
                                                              "linear": "Liniowe", "split": "Split"}, "arithmetic")
            p_cross = real_map[cross_key]
            u_cross_prob = pobierz_parametr("Prawdopodobieństwo krzyżowania", 0.8, float)
            if cross_key in ["arithmetic", "blend", "blend_beta"]: u_alpha = pobierz_parametr("Parametr Alpha", 0.5,
                                                                                              float)
            if cross_key == "blend_beta": u_beta = pobierz_parametr("Parametr Beta", 0.5, float)

            mut_key = pobierz_wybor("Metoda mutacji:", {"gaussian": "Gaussa", "uniform": "Równomierna"}, "gaussian")
            u_mut_prob = pobierz_parametr("Szansa mutacji osobnika", 0.1, float)
            if mut_key == "gaussian":
                p_mut = mutation_gaussian
                u_sigma = pobierz_parametr("Siła mutacji (Sigma)", 1.0, float)
            else:
                p_mut = mutation_uniform
                u_mut_range = pobierz_parametr("Zasięg mutacji (%)", 0.1, float)

    elif typ_rep == "rzeczywista":
        p_cross = crossover_arithmetic
        p_mut = mutation_gaussian

    # --- FITNESS & CALLBACKS ---
    def fitness_func(ga, sol, idx):
        if typ_rep == "binarna":
            decoded = decode_with_binary_chromosome(sol, n_vars, bounds, precision)
        else:
            decoded = sol
        try:
            val = func_obj(decoded)
        except:
            val = func_obj.evaluate(decoded)
        return 1.0 / (np.abs(val) + 1e-8)

    def on_generation(ga_instance):

        pop_fitness = ga_instance.last_generation_fitness
        real_values = [(1.0 / f) - 1e-8 for f in pop_fitness]

        gen = ga_instance.generations_completed
        best = min(real_values)
        avg = np.mean(real_values)
        std = np.std(real_values)

        history['generation'].append(gen)
        history['best_fitness'].append(best)
        history['avg_fitness'].append(avg)
        history['std_fitness'].append(std)

        if gen % 10 == 0 or gen == 1:
            logger.info(f"{gen}: Najlepszy wynik = {best:.6f}")


    print("\n--- START OBLICZEŃ ---")
    ga_instance = pygad.GA(
        num_generations=p_gen, sol_per_pop=p_pop, num_parents_mating=p_par,
        num_genes=num_genes, gene_type=gene_type, gene_space=gene_space,
        init_range_low=init_l, init_range_high=init_h,
        fitness_func=fitness_func,
        parent_selection_type=p_sel,
        K_tournament=u_k_tournament,
        crossover_type=p_cross,
        mutation_type=p_mut,
        keep_elitism=1,
        on_generation=on_generation,
        crossover_probability=u_cross_prob,
        mutation_probability=u_mut_prob
    )


    ga_instance.custom_alpha = u_alpha
    ga_instance.custom_beta = u_beta
    ga_instance.custom_sigma = u_sigma
    ga_instance.custom_mut_range = u_mut_range

    ga_instance.run()


    sol, fit, _ = ga_instance.best_solution()
    dec = decode_with_binary_chromosome(sol, n_vars, bounds, precision) if typ_rep == "binarna" else sol
    final_val = (1.0 / fit) - 1e-8

    print("\n" + "=" * 50)
    print(f"ZAKOŃCZONO: {p_gen} pokoleń")
    print(f"NAJLEPSZY WYNIK: {final_val:.10f}")
    print(f"NAJLEPSZE PARAMETRY: {dec}")
    print("=" * 50)

    plot_results(history, title_suffix=f"(Problem: {wybor_funkcji}, Rep: {typ_rep})")


if __name__ == "__main__":
    main()