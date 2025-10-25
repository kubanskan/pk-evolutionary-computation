import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from .config import GAConfig
from benchmark_functions import Hypersphere, Schwefel, Keane
import numpy as np
import json
from ga.genecticalgorithm import GeneticAlgorithmConfig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .database import DataBase

hypersphere = Hypersphere()
schwefel = Schwefel()
keane = Keane()




BENCHMARK_FUNCTIONS = {
    "Hypersphere": {
        "name": "Hypersphere",
        "bounds": (-5.0, 5.0),
        "optimum": 0,
        "function": hypersphere
    },
    "Schwefel": {
        "name": "Schwefel",
        "bounds": (-500.0, 500.0),
        "optimum": 0,
        "function": schwefel
    },
    "Keane": {
        "name": "Keane",
        "bounds": (0.0, 10.0),
        "optimum": 1.6,
        "function": keane
    },
}


class GeneticAlgorithmGUI:
    """GUI aplikacji algorytmu genetycznego"""

    def __init__(self, root):
        self.root = root
        self.root.title("Projekt 1 – implementacja klasycznego algorytmu genetycznego")
        self.root.geometry("1400x900")

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
        ttk.Entry(params_frame, textvariable=self.precision_var, width=15).grid(row=3, column=1, pady=2)

        ttk.Label(params_frame, text="Rozmiar elity:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.elite_var = tk.IntVar(value=2)
        ttk.Entry(params_frame, textvariable=self.elite_var, width=15).grid(row=4, column=1, pady=2)


        right_frame = ttk.Frame(self.config_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(right_frame, text="Metoda selekcji:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.selection_var = tk.StringVar(value="tournament")
        selection_combo = ttk.Combobox(right_frame, textvariable=self.selection_var,
                                       values=["best", "roulette", "tournament"], state='readonly')
        selection_combo.pack(fill=tk.X, pady=5)

        ttk.Label(right_frame, text="Rozmiar turnieju:").pack(anchor=tk.W)
        self.tournament_var = tk.IntVar(value=3)
        ttk.Entry(right_frame, textvariable=self.tournament_var, width=15).pack(anchor=tk.W, pady=5)

        ttk.Label(right_frame, text="Metoda krzyżowania:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.crossover_method_var = tk.StringVar(value="one_point")
        crossover_combo = ttk.Combobox(right_frame, textvariable=self.crossover_method_var,
                                       values=["one_point", "two_point", "uniform", "discrete"],
                                       state='readonly')
        crossover_combo.pack(fill=tk.X, pady=5)

        ttk.Label(right_frame, text="Prawdopodobieństwo krzyżowania:").pack(anchor=tk.W)
        self.crossover_prob_var = tk.DoubleVar(value=0.8)
        ttk.Entry(right_frame, textvariable=self.crossover_prob_var, width=15).pack(anchor=tk.W, pady=5)

        ttk.Label(right_frame, text="Metoda mutacji:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.mutation_method_var = tk.StringVar(value="one_point")
        mutation_combo = ttk.Combobox(right_frame, textvariable=self.mutation_method_var,
                                      values=["one_point", "two_point", "boundary"],
                                      state='readonly')
        mutation_combo.pack(fill=tk.X, pady=5)

        ttk.Label(right_frame, text="Prawdopodobieństwo mutacji:").pack(anchor=tk.W)
        self.mutation_prob_var = tk.DoubleVar(value=0.01)
        ttk.Entry(right_frame, textvariable=self.mutation_prob_var, width=15).pack(anchor=tk.W, pady=5)

        ttk.Label(right_frame, text="Prawdopodobieństwo inwersji:").pack(anchor=tk.W, pady=(10, 0))
        self.inversion_prob_var = tk.DoubleVar(value=0.05)
        ttk.Entry(right_frame, textvariable=self.inversion_prob_var, width=15).pack(anchor=tk.W, pady=5)

        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(button_frame, text="Uruchom algorytm", command=self.run_algorithm).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Zapisz konfigurację", command=self.save_config).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Wczytaj konfigurację", command=self.load_config).pack(fill=tk.X, pady=5)

    def create_results_tab(self):
        """Tworzenie zakładki wyników"""

        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="Eksportuj wyniki", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Wyczyść", command=lambda: self.results_text.delete(1.0, tk.END)).pack(
            side=tk.LEFT, padx=5)

    def create_plots_tab(self):
        """Tworzenie zakładki wykresów"""

        self.plot_canvas_frame = ttk.Frame(self.plots_frame)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        button_frame = ttk.Frame(self.plots_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="Zapisz wykresy", command=self.save_plots).pack(side=tk.LEFT, padx=5)

    def get_config(self) -> GAConfig:
        """Pobranie konfiguracji z GUI"""
        func_name = self.function_var.get()
        bounds = BENCHMARK_FUNCTIONS[func_name]["bounds"]

        return GAConfig(
            population_size=self.pop_size_var.get(),
            num_generations=self.generations_var.get(),
            num_variables=self.num_vars_var.get(),
            precision=self.precision_var.get(),
            crossover_prob=self.crossover_prob_var.get(),
            mutation_prob=self.mutation_prob_var.get(),
            inversion_prob=self.inversion_prob_var.get(),
            elite_size=self.elite_var.get(),
            selection_method=self.selection_var.get(),
            tournament_size=self.tournament_var.get(),
            crossover_method=self.crossover_method_var.get(),
            mutation_method=self.mutation_method_var.get(),
            bounds=bounds,
            optimization_type=self.opt_type_var.get()
        )

    def run_algorithm(self):
        """Uruchomienie algorytmu genetycznego"""
        try:
            config = self.get_config()
            func_name = self.function_var.get()
            func_info = BENCHMARK_FUNCTIONS[func_name]

            ga = GeneticAlgorithmConfig(config, func_info["function"])

            self.results_text.insert(tk.END, f"Rozpoczęto optymalizację funkcji: {func_info['name']}\n")
            self.results_text.insert(tk.END, f"Liczba zmiennych: {config.num_variables}\n")
            self.results_text.insert(tk.END, f"Typ: {config.optimization_type}\n")
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


            run_id = self.db_manager.save_run(func_name, config.num_variables, result)

            self.update_plots(result['history'])

            messagebox.showinfo("Sukces", "Optymalizacja zakończona pomyślnie!")

        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd: {str(e)}")

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

        filename = filedialog.asksaveasfilename(
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
                'function': self.function_var.get(),
                'optimization_type': self.opt_type_var.get(),
                'population_size': self.pop_size_var.get(),
                'num_generations': self.generations_var.get(),
                'num_variables': self.num_vars_var.get(),
                'precision': self.precision_var.get(),
                'elite_size': self.elite_var.get(),
                'selection_method': self.selection_var.get(),
                'tournament_size': self.tournament_var.get(),
                'crossover_method': self.crossover_method_var.get(),
                'crossover_prob': self.crossover_prob_var.get(),
                'mutation_method': self.mutation_method_var.get(),
                'mutation_prob': self.mutation_prob_var.get(),
                'inversion_prob': self.inversion_prob_var.get()
            }

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

            self.function_var.set(config_dict.get('function', 'sphere'))
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

            messagebox.showinfo("Sukces", "Konfiguracja wczytana!")
