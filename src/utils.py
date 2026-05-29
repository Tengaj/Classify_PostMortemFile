from time import perf_counter
from functools import wraps

def chrono(func):
    """
    Affiche le temps d'éxecution d'une fonction en secondes
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        debut = perf_counter()
        resultat = func(*args, **kwargs)
        fin = perf_counter()
        print(f"L'étape '{func.__name__}' a pris {fin - debut:.4f} secondes")
        return resultat
    return wrapper