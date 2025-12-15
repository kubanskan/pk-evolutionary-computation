import sys
import logging
import pygad
import numpy as np
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
# LOGOWANIE
# ==========================================
logger = logging.getLogger('ga_logger')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('Gen %(message)s'))
    logger.addHandler(ch)


def on_generation(ga_instance):
    if ga_instance.generations_completed % 10 == 0:
        fit = ga_instance.best_solution()[1]
        val = (1.0 / fit) - 1e-8
        logger.info(f"{ga_instance.generations_completed}: Najlepszy wynik = {val:.6f}")


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("   ZAAWANSOWANY KONFIGURATOR ALGORYTMU GENETYCZNEGO")
    print("=" * 60)

    # --- 1. WYBÓR PROBLEMU ---
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
    p_sel = "tournament"
    p_cross = "two_points"
    p_mut = "swap"

    u_k_tournament = 3
    u_cross_prob = 0.8
    u_mut_prob = 0.05
    u_alpha = 0.5
    u_beta = 0.5
    u_sigma = 1.0
    u_mut_range = 0.1

    if adv == 't':
        p_gen = pobierz_parametr("Liczba pokoleń", 50, int)
        p_pop = pobierz_parametr("Rozmiar populacji", 50, int)
        p_par = pobierz_parametr("Liczba rodziców", int(p_pop / 2), int)

        sel_opts = {"tournament": "Turniejowa", "rws": "Koło Ruletki", "random": "Losowa"}
        p_sel = pobierz_wybor("Typ selekcji:", sel_opts, "tournament")
        if p_sel == "tournament":
            u_k_tournament = pobierz_parametr("Rozmiar turnieju (K)", 3, int)

        # --- KONFIGURACJA KRZYŻOWANIA ---
        print("\n   --- Parametry Krzyżowania ---")
        if typ_rep == "binarna":
            p_cross = pobierz_wybor("Metoda:",
                                    {"single_point": "1-pkt", "two_points": "2-pkt", "uniform": "Uniform"},
                                    "two_points")
            u_cross_prob = pobierz_parametr("Prawdopodobieństwo krzyżowania (0.0-1.0)", 0.8, float)
        else:  # Rzeczywista
            real_map = {
                "arithmetic": crossover_arithmetic,
                "blend": crossover_blend_alpha,
                "blend_beta": crossover_blend_alpha_beta,
                "average": crossover_averaging,
                "linear": crossover_linear,
                "split": crossover_simple_split
            }
            cross_key = pobierz_wybor("Metoda:",
                                      {"arithmetic": "Arytmetyczne", "blend": "BLX-Alpha",
                                       "blend_beta": "BLX-Alpha-Beta", "average": "Uśredniające",
                                       "linear": "Liniowe", "split": "Split"},
                                      "arithmetic")
            p_cross = real_map[cross_key]
            u_cross_prob = pobierz_parametr("Prawdopodobieństwo krzyżowania (0.0-1.0)", 0.8, float)

            if cross_key in ["arithmetic", "blend", "blend_beta"]:
                u_alpha = pobierz_parametr(f"Parametr Alpha (dla {cross_key})", 0.5, float)
            if cross_key == "blend_beta":
                u_beta = pobierz_parametr("Parametr Beta", 0.5, float)

        # --- KONFIGURACJA MUTACJI ---
        print("\n   --- Parametry Mutacji ---")
        if typ_rep == "binarna":
            p_mut = pobierz_wybor("Metoda:", {"random": "Random (Flip)", "swap": "Swap"}, "swap")
            u_mut_prob = pobierz_parametr("Prawdopodobieństwo mutacji (0.0-1.0)", 0.05, float)
        else:  # Rzeczywista
            # Wybór między Gaussa a Uniform
            mut_opts = {"gaussian": "Gaussa", "uniform": "Równomierna (Uniform)"}
            mut_key = pobierz_wybor("Metoda mutacji:", mut_opts, "gaussian")

            u_mut_prob = pobierz_parametr("Szansa mutacji osobnika (0.0-1.0)", 0.1, float)

            if mut_key == "gaussian":
                p_mut = mutation_gaussian
                u_sigma = pobierz_parametr("Siła mutacji (Sigma/Odchylenie)", 1.0, float)
            else:
                p_mut = mutation_uniform
                # Zasięg mutacji (jako % zakresu zmiennej)
                u_mut_range = pobierz_parametr("Zasięg mutacji (jako % zakresu, np. 0.1)", 0.1, float)

    elif typ_rep == "rzeczywista":
        p_cross = crossover_arithmetic
        p_mut = mutation_gaussian

    # --- FITNESS ---
    def fitness_bin(ga, sol, idx):
        decoded = decode_with_binary_chromosome(sol, n_vars, bounds, precision)
        try:
            val = func_obj(decoded)
        except:
            val = func_obj.evaluate(decoded)
        return 1.0 / (np.abs(val) + 1e-8)

    def fitness_real(ga, sol, idx):
        try:
            val = func_obj(sol)
        except:
            val = func_obj.evaluate(sol)
        return 1.0 / (np.abs(val) + 1e-8)

    # --- INICJALIZACJA PyGAD ---
    print("\n--- START OBLICZEŃ ---")

    ga_instance = pygad.GA(
        num_generations=p_gen, sol_per_pop=p_pop, num_parents_mating=p_par,
        num_genes=num_genes, gene_type=gene_type, gene_space=gene_space,
        init_range_low=init_l, init_range_high=init_h,
        fitness_func=fitness_bin if typ_rep == "binarna" else fitness_real,
        parent_selection_type=p_sel,
        K_tournament=u_k_tournament,
        crossover_type=p_cross,
        mutation_type=p_mut,
        keep_elitism=1,
        on_generation=on_generation,
        logger=logger,
        crossover_probability=u_cross_prob,
        mutation_probability=u_mut_prob
    )


    ga_instance.custom_alpha = u_alpha
    ga_instance.custom_beta = u_beta
    ga_instance.custom_cross_prob = u_cross_prob
    ga_instance.custom_mut_prob = u_mut_prob
    ga_instance.custom_sigma = u_sigma
    ga_instance.custom_mut_range = u_mut_range

    if adv == 't':
        print(f"(Config: CrossProb={u_cross_prob:.2f}, MutProb={u_mut_prob:.2f})")
        if typ_rep == "rzeczywista":
            if p_mut == mutation_uniform:
                print(f"(Mutation: Uniform, Range={u_mut_range})")
            else:
                print(f"(Mutation: Gaussian, Sigma={u_sigma})")

    # --- URUCHOMIENIE ---
    ga_instance.run()

    # --- WYNIKI ---
    sol, fit, _ = ga_instance.best_solution()
    dec = decode_with_binary_chromosome(sol, n_vars, bounds, precision) if typ_rep == "binarna" else sol
    try:
        final = func_obj(dec)
    except:
        final = func_obj.evaluate(dec)

    print("\n" + "=" * 50)
    print(f"WYNIK: {final:.10f}")
    print(f"PARAMETRY: {dec}")
    print("=" * 50)

    ga_instance.best_solutions_fitness = [(1.0 / x - 1e-8) for x in ga_instance.best_solutions_fitness]
    ga_instance.plot_fitness(title=f"Wynik: {wybor_funkcji}, Rep: {typ_rep}")


if __name__ == "__main__":
    main()