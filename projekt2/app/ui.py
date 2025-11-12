import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from .config import GAConfig
from benchmark_functions import Hypersphere, Schwefel, Keane
from opfunu.cec_based.cec2014 import F12014, F52014, F112014
import numpy as np
import json
from ..ga.genecticalgorithm import GeneticAlgorithmConfig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .database import DataBase
from pathlib import Path


class DynamicBenchmarkFunction:
    """Wrapper dla funkcji benchmark z dynamiczną zmianą wymiaru"""

    def __init__(self, benchmark_class, name):
        self.benchmark_class = benchmark_class
        self.name = name
        self.func_instance = None
        self.current_ndim = None
        self.bounds = None
        self.optimum = None

    def _initialize(self, ndim):
        """Inicjalizacja funkcji z danym wymiarem"""
        if ndim != self.current_ndim:
            self.current_ndim = ndim
            self.func_instance = self.benchmark_class(n_dimensions=ndim)
            bounds_tuple = self.func_instance.suggested_bounds()
            self.bounds = (float(bounds_tuple[0][0]), float(bounds_tuple[1][0]))
            minima_data = self.func_instance.minima()
            if minima_data and len(minima_data) > 0:
                self.optimum = float(minima_data[0].score)
            else:
                self.optimum = 0.0

    def __call__(self, x):
        """Ewaluacja funkcji"""
        x = np.asarray(x)
        self._initialize(len(x))
        return self.func_instance(x)


class DynamicCECFunction:
    """Wrapper dla funkcji CEC z dynamiczną zmianą wymiaru"""

    def __init__(self, cec_class, name, initial_ndim=10):
        self.cec_class = cec_class
        self.name = name
        self.current_ndim = initial_ndim
        self.func_instance = None
        self._initialize(initial_ndim)

    def _initialize(self, ndim):
        """Inicjalizacja funkcji z danym wymiarem"""
        self.current_ndim = ndim
        self.func_instance = self.cec_class(ndim=ndim)
        self.bounds = (float(self.func_instance.lb[0]), float(self.func_instance.ub[0]))
        self.optimum = float(self.func_instance.f_global)

    def update_dimension(self, ndim):
        """Aktualizuj wymiar funkcji"""
        if ndim != self.current_ndim:
            self._initialize(ndim)

    def __call__(self, x):
        """Ewaluacja funkcji"""
        x = np.asarray(x)
        if len(x) != self.current_ndim:
            self._initialize(len(x))
        return self.func_instance.evaluate(x)


dynamic_hypersphere = DynamicBenchmarkFunction(Hypersphere, "Hypersphere")
dynamic_schwefel = DynamicBenchmarkFunction(Schwefel, "Schwefel")
dynamic_keane = DynamicBenchmarkFunction(Keane, "Keane")

cec_f1 = DynamicCECFunction(F12014, "CEC2014 F1: Rotated High Conditioned Elliptic")
cec_f5 = DynamicCECFunction(F52014, "CEC2014 F5: Shifted and Rotated Ackley")
cec_f11 = DynamicCECFunction(F112014, "CEC2014 F11: Shifted and Rotated Schwefel")

BENCHMARK_FUNCTIONS = {
    "Hypersphere": {
        "name": "Hypersphere",
        "bounds": (-5.0, 5.0),
        "optimum": 0.0,
        "function": dynamic_hypersphere,
        "is_cec": False
    },
    "Schwefel": {
        "name": "Schwefel",
        "bounds": (-500.0, 500.0),
        "optimum": 0.0,
        "function": dynamic_schwefel,
        "is_cec": False
    },
    "Keane": {
        "name": "Keane",
        "bounds": (0.0, 10.0),
        "optimum": 1.6,
        "function": dynamic_keane,
        "is_cec": False
    },
    "CEC2014-F1": {
        "name": "CEC2014 F1: Rotated High Conditioned Elliptic",
        "bounds": cec_f1.bounds,
        "optimum": cec_f1.optimum,
        "function": cec_f1,
        "is_cec": True
    },
    "CEC2014-F5": {
        "name": "CEC2014 F5: Shifted and Rotated Ackley",
        "bounds": cec_f5.bounds,
        "optimum": cec_f5.optimum,
        "function": cec_f5,
        "is_cec": True
    },
    "CEC2014-F11": {
        "name": "CEC2014 F11: Shifted and Rotated Schwefel",
        "bounds": cec_f11.bounds,
        "optimum": cec_f11.optimum,
        "function": cec_f11,
        "is_cec": True
    },
}


class GeneticAlgorithmGUI:
    """GUI aplikacji algorytmu genetycznego"""

    def __init__(self, root):
        self.root = root
        self.root.title("Projekt 2 – Reprezentacja binarna vs. rzeczywista")
        self.root.geometry("1400x950")

        self.db_manager = DataBase()
        self.current_result = None

        self.create_widgets()

    def create_widgets(self):
        """Tworzenie widgetów GUI"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Konfiguracja")
        self.create_config_tab()

        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Wyniki")
        self.create_results_tab()

        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text="Wykresy")
        self.create_plots_tab()

    def create_config_tab(self):
        """Tworzenie zakładki konfiguracji"""
        left_frame = ttk.Frame(self.config_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        repr_frame = ttk.LabelFrame(left_frame, text="Reprezentacja chromosomu", padding=10)
        repr_frame.pack(fill=tk.X, pady=(0, 10))

        self.representation_var = tk.StringVar(value="binary")
        ttk.Radiobutton(repr_frame, text="Binarna", variable=self.representation_var,
                        value="binary", command=self.on_representation_change).pack(anchor=tk.W)
        ttk.Radiobutton(repr_frame, text="Rzeczywista", variable=self.representation_var,
                        value="real", command=self.on_representation_change).pack(anchor=tk.W)

        ttk.Label(left_frame, text="Funkcja testowa:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.function_var = tk.StringVar(value="Hypersphere")
        function_combo = ttk.Combobox(left_frame, textvariable=self.function_var,
                                      values=list(BENCHMARK_FUNCTIONS.keys()), state='readonly')
        function_combo.pack(fill=tk.X, pady=5)

        ttk.Label(left_frame, text="Typ optymalizacji:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.opt_type_var = tk.StringVar(value="minimize")
        ttk.Radiobutton(left_frame, text="Minimalizacja", variable=self.opt_type_var, value="minimize").pack(
            anchor=tk.W)
        ttk.Radiobutton(left_frame, text="Maksymalizacja", variable=self.opt_type_var, value="maximize").pack(
            anchor=tk.W)

        ttk.Label(left_frame, text="Parametry podstawowe:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        params_frame = ttk.Frame(left_frame)
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text="Rozmiar populacji:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pop_size_var = tk.IntVar(value=100)
        ttk.Entry(params_frame, textvariable=self.pop_size_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(params_frame, text="Liczba generacji:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.generations_var = tk.IntVar(value=100)
        ttk.Entry(params_frame, textvariable=self.generations_var, width=15).grid(row=1, column=1, pady=2)

        ttk.Label(params_frame, text="Liczba zmiennych:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.num_vars_var = tk.IntVar(value=10)
        ttk.Entry(params_frame, textvariable=self.num_vars_var, width=15).grid(row=2, column=1, pady=2)

        ttk.Label(params_frame, text="Precyzja (bity):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.precision_var = tk.IntVar(value=16)
        self.precision_entry = ttk.Entry(params_frame, textvariable=self.precision_var, width=15)
        self.precision_entry.grid(row=3, column=1, pady=2)

        ttk.Label(params_frame, text="Liczba elit:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.elite_var = tk.IntVar(value=2)
        ttk.Entry(params_frame, textvariable=self.elite_var, width=15).grid(row=4, column=1, pady=2)

        right_frame = ttk.Frame(self.config_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.selection_frame = ttk.LabelFrame(right_frame, text="Selekcja", padding=10)
        self.selection_frame.pack(fill=tk.X, pady=(0, 10))

        self.selection_var = tk.StringVar(value="tournament")
        ttk.Radiobutton(self.selection_frame, text="Najlepsi", variable=self.selection_var,
                        value="best", command=self.on_selection_change).pack(anchor=tk.W)
        ttk.Radiobutton(self.selection_frame, text="Ruletka", variable=self.selection_var,
                        value="roulette", command=self.on_selection_change).pack(anchor=tk.W)
        ttk.Radiobutton(self.selection_frame, text="Turniejowa", variable=self.selection_var,
                        value="tournament", command=self.on_selection_change).pack(anchor=tk.W)

        self.selection_params_frame = ttk.Frame(self.selection_frame)
        self.selection_params_frame.pack(fill=tk.X, pady=5)

        self.tournament_var = tk.IntVar(value=3)
        self.selection_percentage_var = tk.DoubleVar(value=0.5)

        self.on_selection_change()

        self.crossover_frame = ttk.LabelFrame(right_frame, text="Krzyżowanie", padding=10)
        self.crossover_frame.pack(fill=tk.X, pady=(0, 10))

        self.crossover_method_var = tk.StringVar(value="one_point")
        self.crossover_prob_var = tk.DoubleVar(value=0.8)

        self.create_crossover_widgets()

        self.mutation_frame = ttk.LabelFrame(right_frame, text="Mutacja", padding=10)
        self.mutation_frame.pack(fill=tk.X, pady=(0, 10))

        self.mutation_method_var = tk.StringVar(value="one_point")
        self.mutation_prob_var = tk.DoubleVar(value=0.01)

        self.create_mutation_widgets()

        self.inversion_frame = ttk.LabelFrame(right_frame, text="Inwersja (tylko binarna)", padding=10)
        self.inversion_frame.pack(fill=tk.X, pady=(0, 10))

        inv_params = ttk.Frame(self.inversion_frame)
        inv_params.pack(fill=tk.X)
        ttk.Label(inv_params, text="Prawdopodobieństwo:").grid(row=0, column=0, sticky=tk.W)
        self.inversion_prob_var = tk.DoubleVar(value=0.05)
        ttk.Entry(inv_params, textvariable=self.inversion_prob_var, width=10).grid(row=0, column=1)

        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        ttk.Button(buttons_frame, text="Uruchom GA", command=self.run_ga).pack(side=tk.LEFT, padx=5)

        self.on_representation_change()

    def on_representation_change(self):
        """Callback wywoływany przy zmianie reprezentacji"""
        representation = self.representation_var.get()

        if representation == "binary":
            self.precision_entry.config(state='normal')
            self.inversion_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.precision_entry.config(state='disabled')
            self.inversion_frame.pack_forget()

        self.create_crossover_widgets()
        self.create_mutation_widgets()

    def create_crossover_widgets(self):
        """Dynamiczne tworzenie widgetów krzyżowania"""

        for widget in self.crossover_frame.winfo_children():
            widget.destroy()

        representation = self.representation_var.get()

        prob_frame = ttk.Frame(self.crossover_frame)
        prob_frame.pack(fill=tk.X, pady=5)
        ttk.Label(prob_frame, text="Prawdopodobieństwo:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(prob_frame, textvariable=self.crossover_prob_var, width=10).grid(row=0, column=1)

        ttk.Label(self.crossover_frame, text="Metoda:").pack(anchor=tk.W)

        if representation == "binary":
            methods = [
                ("Jednopunktowe", "one_point"),
                ("Dwupunktowe", "two_point"),
                ("Jednorodne", "uniform"),
                ("Ziarniste", "discrete")
            ]
        else:
            methods = [
                ("Arytmetyczne", "arithmetic"),
                ("Liniowe", "linear"),
                ("Mieszające alfa (BLX-α)", "blend_alpha"),
                ("Mieszające alfa-beta", "blend_alpha_beta"),
                ("Uśredniające", "averaging")
            ]

        for label, value in methods:
            ttk.Radiobutton(self.crossover_frame, text=label,
                            variable=self.crossover_method_var, value=value,
                            command=self.on_crossover_change).pack(anchor=tk.W)

        if representation == "real":
            self.crossover_params_frame = ttk.Frame(self.crossover_frame)
            self.crossover_params_frame.pack(fill=tk.X, pady=5)

            self.arithmetic_alpha_var = tk.DoubleVar(value=0.5)
            self.blend_alpha_var = tk.DoubleVar(value=0.5)
            self.blend_alpha_param_var = tk.DoubleVar(value=0.5)
            self.blend_beta_param_var = tk.DoubleVar(value=0.5)

            self.on_crossover_change()

    def on_crossover_change(self):
        """Callback wywoływany przy zmianie metody krzyżowania"""
        if self.representation_var.get() != "real":
            return

        for widget in self.crossover_params_frame.winfo_children():
            widget.destroy()

        method = self.crossover_method_var.get()

        if method == "arithmetic":
            ttk.Label(self.crossover_params_frame, text="Alpha:").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.crossover_params_frame, textvariable=self.arithmetic_alpha_var,
                      width=10).grid(row=0, column=1)
        elif method == "blend_alpha":
            ttk.Label(self.crossover_params_frame, text="Alpha:").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.crossover_params_frame, textvariable=self.blend_alpha_var,
                      width=10).grid(row=0, column=1)
        elif method == "blend_alpha_beta":
            ttk.Label(self.crossover_params_frame, text="Alpha:").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.crossover_params_frame, textvariable=self.blend_alpha_param_var,
                      width=10).grid(row=0, column=1)
            ttk.Label(self.crossover_params_frame, text="Beta:").grid(row=1, column=0, sticky=tk.W)
            ttk.Entry(self.crossover_params_frame, textvariable=self.blend_beta_param_var,
                      width=10).grid(row=1, column=1)

    def create_mutation_widgets(self):
        """Dynamiczne tworzenie widgetów mutacji"""

        for widget in self.mutation_frame.winfo_children():
            widget.destroy()

        representation = self.representation_var.get()

        prob_frame = ttk.Frame(self.mutation_frame)
        prob_frame.pack(fill=tk.X, pady=5)
        ttk.Label(prob_frame, text="Prawdopodobieństwo:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(prob_frame, textvariable=self.mutation_prob_var, width=10).grid(row=0, column=1)

        ttk.Label(self.mutation_frame, text="Metoda:").pack(anchor=tk.W)

        if representation == "binary":
            methods = [
                ("Jednopunktowa", "one_point"),
                ("Dwupunktowa", "two_point"),
                ("Brzegowa", "boundary")
            ]
        else:
            methods = [
                ("Równomierna", "uniform"),
                ("Gaussa", "gaussian")
            ]

        for label, value in methods:
            ttk.Radiobutton(self.mutation_frame, text=label,
                            variable=self.mutation_method_var, value=value,
                            command=self.on_mutation_change).pack(anchor=tk.W)

        if representation == "real":
            self.mutation_params_frame = ttk.Frame(self.mutation_frame)
            self.mutation_params_frame.pack(fill=tk.X, pady=5)

            self.mutation_range_var = tk.DoubleVar(value=0.1)
            self.gaussian_sigma_var = tk.DoubleVar(value=0.1)

            self.on_mutation_change()

    def on_mutation_change(self):
        """Callback wywoływany przy zmianie metody mutacji"""
        if self.representation_var.get() != "real":
            return

        for widget in self.mutation_params_frame.winfo_children():
            widget.destroy()

        method = self.mutation_method_var.get()

        if method == "uniform":
            ttk.Label(self.mutation_params_frame, text="Zakres (% range):").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.mutation_params_frame, textvariable=self.mutation_range_var,
                      width=10).grid(row=0, column=1)
        elif method == "gaussian":
            ttk.Label(self.mutation_params_frame, text="Sigma (% range):").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.mutation_params_frame, textvariable=self.gaussian_sigma_var,
                      width=10).grid(row=0, column=1)

    def on_selection_change(self):
        """Callback wywoływany przy zmianie metody selekcji"""

        for widget in self.selection_params_frame.winfo_children():
            widget.destroy()

        method = self.selection_var.get()

        if method == "tournament":
            ttk.Label(self.selection_params_frame, text="Rozmiar turnieju:").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.selection_params_frame, textvariable=self.tournament_var,
                      width=10).grid(row=0, column=1)
        elif method == "best":
            ttk.Label(self.selection_params_frame, text="Procent najlepszych (0-1):").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.selection_params_frame, textvariable=self.selection_percentage_var,
                      width=10).grid(row=0, column=1)

            ttk.Entry(self.mutation_params_frame, textvariable=self.mutation_range_var,
                      width=10).grid(row=0, column=1)
        elif method == "gaussian":
            ttk.Label(self.mutation_params_frame, text="Sigma (% range):").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(self.mutation_params_frame, textvariable=self.gaussian_sigma_var,
                      width=10).grid(row=0, column=1)

    def create_results_tab(self):
        """Tworzenie zakładki wyników"""
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, height=30, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        buttons_frame = ttk.Frame(self.results_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(buttons_frame, text="Eksportuj wyniki", command=self.export_results).pack(side=tk.LEFT, padx=5)

    def create_plots_tab(self):
        """Tworzenie zakładki wykresów"""
        self.plot_canvas_frame = ttk.Frame(self.plots_frame)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        buttons_frame = ttk.Frame(self.plots_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(buttons_frame, text="Zapisz wykresy", command=self.save_plots).pack(side=tk.LEFT, padx=5)

    def get_config(self) -> GAConfig:
        """Pobierz konfigurację z GUI"""
        representation = self.representation_var.get()

        config = GAConfig(
            population_size=self.pop_size_var.get(),
            num_generations=self.generations_var.get(),
            num_variables=self.num_vars_var.get(),
            precision=self.precision_var.get(),
            representation=representation,
            elite_size=self.elite_var.get(),
            selection_method=self.selection_var.get(),
            tournament_size=self.tournament_var.get(),
            crossover_method=self.crossover_method_var.get(),
            crossover_prob=self.crossover_prob_var.get(),
            mutation_method=self.mutation_method_var.get(),
            mutation_prob=self.mutation_prob_var.get(),
            inversion_prob=self.inversion_prob_var.get(),
            optimization_type=self.opt_type_var.get()
        )
        if representation == "real":
            if hasattr(self, 'arithmetic_alpha_var'):
                config.arithmetic_alpha = self.arithmetic_alpha_var.get()
            if hasattr(self, 'blend_alpha_var'):
                config.blend_alpha = self.blend_alpha_var.get()
            if hasattr(self, 'blend_alpha_param_var'):
                config.blend_alpha_param = self.blend_alpha_param_var.get()
                config.blend_beta_param = self.blend_beta_param_var.get()
            if hasattr(self, 'mutation_range_var'):
                config.mutation_range = self.mutation_range_var.get()
            if hasattr(self, 'gaussian_sigma_var'):
                config.gaussian_sigma = self.gaussian_sigma_var.get()

        return config

    def run_ga(self):
        """Uruchomienie algorytmu genetycznego"""
        try:
            config = self.get_config()

            func_name = self.function_var.get()
            func_info = BENCHMARK_FUNCTIONS[func_name].copy()
            func = func_info["function"]

            config.bounds = func_info["bounds"]

            if isinstance(func, DynamicBenchmarkFunction):
                test_x = np.zeros(config.num_variables)
                func(test_x)
                func_info["bounds"] = func.bounds
                func_info["optimum"] = func.optimum
            elif isinstance(func, DynamicCECFunction):
                func.update_dimension(config.num_variables)
                func_info["bounds"] = func.bounds
                func_info["optimum"] = func.optimum

            ga = GeneticAlgorithmConfig(config, func)

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Rozpoczęto optymalizację funkcji: {func_info['name']}\n")
            self.results_text.insert(tk.END, f"Reprezentacja: {config.representation.upper()}\n")
            self.results_text.insert(tk.END, f"Liczba zmiennych: {config.num_variables}\n")
            self.results_text.insert(tk.END, f"Typ: {config.optimization_type}\n")

            if config.representation == "binary":
                self.results_text.insert(tk.END, f"Precyzja: {config.precision} bitów\n")

            self.results_text.insert(tk.END, f"Metoda krzyżowania: {config.crossover_method}\n")
            self.results_text.insert(tk.END, f"Metoda mutacji: {config.mutation_method}\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n")
            self.results_text.update()

            result = ga.evolve()
            self.current_result = result

            self.results_text.insert(tk.END, f"\nWyniki optymalizacji:\n")
            self.results_text.insert(tk.END, f"Czas wykonania: {result['elapsed_time']:.4f} sekund\n")
            self.results_text.insert(tk.END, f"Najlepsza wartość funkcji: {result['best_fitness']:.10f}\n")
            self.results_text.insert(tk.END, f"Wartość optimum: {func_info['optimum']}\n")
            self.results_text.insert(tk.END, f"Błąd: {abs(result['best_fitness'] - func_info['optimum']):.10f}\n")
            self.results_text.insert(tk.END, f"\nNajlepsze rozwiązanie (pierwsze 5 zmiennych):\n")
            for i, val in enumerate(result['best_solution'][:5]):
                self.results_text.insert(tk.END, f"  x[{i}] = {val:.6f}\n")
            if len(result['best_solution']) > 5:
                self.results_text.insert(tk.END, f"  ... (pozostałe {len(result['best_solution']) - 5} zmiennych)\n")

            self.results_text.insert(tk.END, "\n" + "*" * 60 + "\n\n")

            run_id = self.db_manager.save_run(f"{func_name}_{config.representation}",
                                              config.num_variables, result)

            self.update_plots(result['history'])

            messagebox.showinfo("Sukces", "Optymalizacja zakończona pomyślnie!")

        except Exception as e:
            import traceback
            messagebox.showerror("Błąd", f"Wystąpił błąd: {str(e)}\n\n{traceback.format_exc()}")

    def update_plots(self, history: dict):
        """Aktualizacja wykresów"""
        for widget in self.plot_canvas_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(history['generation'], history['best_fitness'], 'b-', linewidth=2, label='Najlepsza')
        axes[0].set_xlabel('Generacja')
        axes[0].set_ylabel('Wartość funkcji')
        axes[0].set_title('Wartość funkcji celu od kolejnej iteracji')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        generations = np.array(history['generation'])
        avg = np.array(history['avg_fitness'])
        std = np.array(history['std_fitness'])

        axes[1].plot(generations, avg, 'g-', linewidth=2, label='Średnia')
        axes[1].fill_between(generations, avg - std, avg + std, alpha=0.3, color='green', label='±1 std')
        axes[1].set_xlabel('Generacja')
        axes[1].set_ylabel('Wartość funkcji')
        axes[1].set_title('Średnia wartość funkcji i odchylenie standardowe')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_plots(self):
        """Zapisanie wykresów do pliku"""
        if self.current_result is None:
            messagebox.showwarning("Brak danych", "Najpierw uruchom algorytm!")
            return

        downloads_path = str(Path.home() / "Downloads")
        filename = filedialog.asksaveasfilename(
            initialdir=downloads_path,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if filename:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            history = self.current_result['history']

            axes[0].plot(history['generation'], history['best_fitness'], 'b-', linewidth=2)
            axes[0].set_xlabel('Generacja')
            axes[0].set_ylabel('Wartość funkcji')
            axes[0].set_title('Wartość funkcji celu od kolejnej iteracji')
            axes[0].grid(True, alpha=0.3)

            generations = np.array(history['generation'])
            avg = np.array(history['avg_fitness'])
            std = np.array(history['std_fitness'])

            axes[1].plot(generations, avg, 'g-', linewidth=2, label='Średnia')
            axes[1].fill_between(generations, avg - std, avg + std, alpha=0.3, color='green')
            axes[1].set_xlabel('Generacja')
            axes[1].set_ylabel('Wartość funkcji')
            axes[1].set_title('Średnia wartość funkcji i odchylenie standardowe')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            messagebox.showinfo("Sukces", f"Wykresy zapisane do: {filename}")

    def export_results(self):
        """Eksport wyników do pliku"""
        if self.current_result is None:
            messagebox.showwarning("Brak danych", "Najpierw uruchom algorytm!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            content = self.results_text.get(1.0, tk.END)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Sukces", f"Wyniki zapisane do: {filename}")

    def save_config(self):
        """Zapisanie konfiguracji do pliku"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            config = self.get_config()
            config_dict = {
                'representation': config.representation,
                'function': self.function_var.get(),
                'optimization_type': config.optimization_type,
                'population_size': config.population_size,
                'num_generations': config.num_generations,
                'num_variables': config.num_variables,
                'precision': config.precision,
                'elite_size': config.elite_size,
                'selection_method': config.selection_method,
                'tournament_size': config.tournament_size,
                'crossover_method': config.crossover_method,
                'crossover_prob': config.crossover_prob,
                'mutation_method': config.mutation_method,
                'mutation_prob': config.mutation_prob,
                'inversion_prob': config.inversion_prob
            }

            if config.representation == "real":
                config_dict.update({
                    'arithmetic_alpha': config.arithmetic_alpha,
                    'blend_alpha': config.blend_alpha,
                    'blend_alpha_param': config.blend_alpha_param,
                    'blend_beta_param': config.blend_beta_param,
                    'mutation_range': config.mutation_range,
                    'gaussian_sigma': config.gaussian_sigma
                })

            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=4)

            messagebox.showinfo("Sukces", f"Konfiguracja zapisana do: {filename}")

    def load_config(self):
        """Wczytanie konfiguracji z pliku"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            with open(filename, 'r') as f:
                config_dict = json.load(f)

            self.representation_var.set(config_dict.get('representation', 'binary'))
            self.function_var.set(config_dict.get('function', 'Hypersphere'))
            self.opt_type_var.set(config_dict.get('optimization_type', 'minimize'))
            self.pop_size_var.set(config_dict.get('population_size', 100))
            self.generations_var.set(config_dict.get('num_generations', 100))
            self.num_vars_var.set(config_dict.get('num_variables', 10))
            self.precision_var.set(config_dict.get('precision', 16))
            self.elite_var.set(config_dict.get('elite_size', 2))
            self.selection_var.set(config_dict.get('selection_method', 'tournament'))
            self.tournament_var.set(config_dict.get('tournament_size', 3))
            self.crossover_method_var.set(config_dict.get('crossover_method', 'one_point'))
            self.crossover_prob_var.set(config_dict.get('crossover_prob', 0.8))
            self.mutation_method_var.set(config_dict.get('mutation_method', 'one_point'))
            self.mutation_prob_var.set(config_dict.get('mutation_prob', 0.01))
            self.inversion_prob_var.set(config_dict.get('inversion_prob', 0.05))

            if config_dict.get('representation') == 'real':
                if 'arithmetic_alpha' in config_dict:
                    self.arithmetic_alpha_var.set(config_dict['arithmetic_alpha'])
                if 'blend_alpha' in config_dict:
                    self.blend_alpha_var.set(config_dict['blend_alpha'])
                if 'blend_alpha_param' in config_dict:
                    self.blend_alpha_param_var.set(config_dict['blend_alpha_param'])
                    self.blend_beta_param_var.set(config_dict['blend_beta_param'])
                if 'mutation_range' in config_dict:
                    self.mutation_range_var.set(config_dict['mutation_range'])
                if 'gaussian_sigma' in config_dict:
                    self.gaussian_sigma_var.set(config_dict['gaussian_sigma'])

            self.on_representation_change()
            messagebox.showinfo("Sukces", "Konfiguracja wczytana!")
