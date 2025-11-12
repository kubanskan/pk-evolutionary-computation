import tkinter as tk
from .app.ui import GeneticAlgorithmGUI
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Uruchomienie GUI aplikacji"""
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

