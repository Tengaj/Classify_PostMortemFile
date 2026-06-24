import yaml
from pathlib import Path
from time import perf_counter
from functools import wraps

def _load_config(filename="config.yaml"):
    """
    Charge le fichier de configuration YAML depuis la racine du projet.
    """
    # __file__ est le chemin de utils.py
    # .resolve().parent.parent remonte de 'src/' vers la racine du projet
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / filename
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {config_path} est introuvable.")
        return None

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

CONFIG = _load_config()