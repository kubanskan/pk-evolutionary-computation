import numpy as np
from chromosome import BinaryChromosome

def decode_with_binary_chromosome(solution, n_variables, bounds, precision):
    """
    Łączy PyGAD (który widzi tylko 0 i 1) z funkcją celu (która potrzebuje liczb rzeczywistych)
    """

    chrom = BinaryChromosome(n_variables, bounds, precision)
    chrom.genes = np.array(solution, dtype=int)
    return chrom.decode()