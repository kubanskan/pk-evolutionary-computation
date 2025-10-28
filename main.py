import tkinter as tk
from app.ui import GeneticAlgorithmGUI

def main():
    """Uruchomienie GUI aplikacji"""
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()