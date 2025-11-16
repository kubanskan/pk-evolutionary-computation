import tkinter as tk
from projekt2.app.ui import GeneticAlgorithmGUI

def main():
    """Uruchomienie GUI aplikacji"""
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

