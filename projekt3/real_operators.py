import numpy as np


# ==========================================
# NARZĘDZIA POMOCNICZE
# ==========================================

def _get_param(ga_instance, name, default):
    """Bezpieczne pobieranie parametru z instancji PyGAD."""
    return getattr(ga_instance, name, default)


def _should_cross(ga_instance):
    """
    Decyzja czy krzyżować parę rodziców.
    Pobiera 'custom_cross_prob' (domyślnie 1.0 - zawsze).
    """
    prob = _get_param(ga_instance, "custom_cross_prob", 1.0)
    return np.random.rand() <= prob


# ==========================================
# OPERATORY KRZYŻOWANIA
# ==========================================

def crossover_arithmetic(parents, offspring_size, ga_instance):
    """
    Krzyżowanie arytmetyczne.
    Parametry: Alpha, Probability.
    """
    offspring = np.empty(offspring_size)
    idx = 0
    alpha = _get_param(ga_instance, "custom_alpha", 0.5)

    while idx < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        if _should_cross(ga_instance):
            offspring[idx] = alpha * parent1 + (1 - alpha) * parent2
        else:
            offspring[idx] = parent1
        idx += 1
    return offspring


def crossover_blend_alpha(parents, offspring_size, ga_instance):
    """
    BLX-Alpha.
    Parametry: Alpha, Probability.
    """
    offspring = np.empty(offspring_size)
    idx = 0
    alpha = _get_param(ga_instance, "custom_alpha", 0.5)

    while idx < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        if _should_cross(ga_instance):
            min_v = np.minimum(parent1, parent2)
            max_v = np.maximum(parent1, parent2)
            diff = max_v - min_v

            lower = min_v - alpha * diff
            upper = max_v + alpha * diff
            offspring[idx] = np.random.uniform(lower, upper)
        else:
            offspring[idx] = parent1
        idx += 1
    return offspring


def crossover_blend_alpha_beta(parents, offspring_size, ga_instance):
    """
    BLX-Alpha-Beta.
    Parametry: Alpha, Beta, Probability.
    """
    offspring = np.empty(offspring_size)
    idx = 0
    alpha = _get_param(ga_instance, "custom_alpha", 0.5)
    beta = _get_param(ga_instance, "custom_beta", 0.5)

    while idx < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        if _should_cross(ga_instance):
            min_v = np.minimum(parent1, parent2)
            max_v = np.maximum(parent1, parent2)
            diff = max_v - min_v

            lower = min_v - alpha * diff
            upper = max_v + beta * diff
            offspring[idx] = np.random.uniform(lower, upper)
        else:
            offspring[idx] = parent1
        idx += 1
    return offspring


def crossover_linear(parents, offspring_size, ga_instance):
    """
    Krzyżowanie Liniowe (Wybór najlepszego z 3).
    Parametry: Probability.
    """
    offspring = np.empty(offspring_size)
    idx = 0
    fitness_func = ga_instance.fitness_func

    while idx < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        if _should_cross(ga_instance):
            c1 = 0.5 * parent1 + 0.5 * parent2
            c2 = 1.5 * parent1 - 0.5 * parent2
            c3 = -0.5 * parent1 + 1.5 * parent2
            candidates = [c1, c2, c3]

            scores = []
            for cand in candidates:
                try:
                    score = fitness_func(ga_instance, cand, 0)
                except:
                    score = -float('inf')
                scores.append(score)


            best_idx = np.argmax(scores)
            offspring[idx] = candidates[best_idx]
        else:
            offspring[idx] = parent1
        idx += 1
    return offspring


def crossover_averaging(parents, offspring_size, ga_instance):
    """Uśredniające. Parametry: Probability."""
    offspring = np.empty(offspring_size)
    idx = 0
    while idx < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        if _should_cross(ga_instance):
            offspring[idx] = (parent1 + parent2) / 2.0
        else:
            offspring[idx] = parent1
        idx += 1
    return offspring


def crossover_simple_split(parents, offspring_size, ga_instance):
    """Split. Parametry: Probability."""
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        if _should_cross(ga_instance):
            if offspring_size[1] > 1:
                pt = np.random.choice(range(offspring_size[1]))
                parent1[pt:] = parent2[pt:]
        offspring.append(parent1)
        idx += 1
    return np.array(offspring)


# ==========================================
# MUTACJA
# ==========================================

def mutation_gaussian(offspring, ga_instance):
    """
    Mutacja Gaussa.
    Parametry:
      - custom_mut_prob: Prawdopodobieństwo mutacji pojedynczego GENU.
      - custom_sigma: Odchylenie standardowe (siła mutacji).
    """
    sigma = _get_param(ga_instance, "custom_sigma", 1.0)
    prob = _get_param(ga_instance, "custom_mut_prob", 0.1)

    for chromosome_idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.rand() <= prob:
                noise = np.random.normal(0, sigma)
                offspring[chromosome_idx, gene_idx] += noise

    return offspring


def mutation_uniform(offspring, ga_instance):
    """
    Mutacja Równomierna (Uniform).
    Zmienia JEDEN losowy gen w chromosomie o wartość z zakresu [-range, +range].
    """
    prob = _get_param(ga_instance, "custom_mut_prob", 0.1)
    mut_range_coeff = _get_param(ga_instance, "custom_mut_range", 0.1)  # Np. 0.1 to 10% zakresu

    low = ga_instance.init_range_low
    high = ga_instance.init_range_high
    var_range = abs(high - low)

    for chromosome_idx in range(offspring.shape[0]):
        if np.random.rand() <= prob:
            gene_idx = np.random.randint(0, offspring.shape[1])

            delta = np.random.uniform(-mut_range_coeff * var_range,
                                      mut_range_coeff * var_range)

            offspring[chromosome_idx, gene_idx] += delta

    return offspring