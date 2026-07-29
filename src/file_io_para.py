import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import chrono, CONFIG
from classification import classify_file

EXEC_CFG = CONFIG.get("execution", {})
DEFAULT_INPUT_PATH = EXEC_CFG.get("input_path", "../../LLRF/")
DEFAULT_OUTPUT_PATH = EXEC_CFG.get("output_path", "../resultat/")

TARGET_FAULTS = EXEC_CFG.get("target_faults", [
    "Rég signal RF hors tolérance", 
    "Dép seuil de sécurité RF", 
    "Seuil de vide", 
    "Claquage ou quench cavité"
])

def _process_file_worker(file_path):
    """
    Fonction 'ouvrier' exécutée en parallèle. 
    Elle s'occupe d'un seul fichier et renvoie le résultat et le message à afficher.
    """
    if file_path.stat().st_size == 0:
        return file_path, None, f"Fichier ignoré : {file_path.name} (0 octet)"

    try:
        label = classify_file(file_path)
        fault = label.split(";")[0]
        
        if fault in TARGET_FAULTS:
            msg = f"Nom : {file_path.name} | Défaut : {label}"
            return file_path, label, msg
        else:
            msg = f"Nom : {file_path.name} | Défaut : {label}"
            return file_path, None, msg
            
    except Exception as e:
        return file_path, None, f"Fichier ignoré ({file_path.name}) - Erreur : {e}"

def _run_parallel(files_list):
    """
    Moteur de parallélisation. Distribue la liste de fichiers aux cœurs du CPU.
    """
    results = {}
    
    # max_workers par défaut utilise 100% des cœurs de ton processeur
    with ProcessPoolExecutor() as executor:
        # On lance tous les calculs en parallèle
        future_to_file = {executor.submit(_process_file_worker, f): f for f in files_list}

        # as_completed permet de traiter les résultats dès qu'un processus a fini
        for future in as_completed(future_to_file):
            file_path, label, msg = future.result()
            
            #print(msg) 
            
            if label is not None:
                results[file_path] = label
                
    return results

def classify_folder(folder_name):
    """
    Classifie tous les fichiers d'un dossier en parallèle.
    """
    folder_path = Path(folder_name)
    files_list = [path for path in folder_path.iterdir() if path.is_file()]
    
    return _run_parallel(files_list)

def classify_cav_folder(cavity_folder_name):
    """
    Classifie tous les fichiers des sous-dossiers d'un type de cavité en parallèle.
    """
    folder_path = Path(cavity_folder_name)
    subfolders = [path for path in folder_path.iterdir() if path.is_dir()]
    
    files_list = []
    for sub in subfolders:
        files_list.extend([p for p in sub.iterdir() if p.is_file()])
        
    return _run_parallel(files_list)

@chrono
def classify_year_folder(year_folder_name, export_prefix='', save_csv=True, output_path=DEFAULT_OUTPUT_PATH):
    """
    Extrait tous les fichiers d'une année et les classifie en parallèle.
    Si save_csv est False, la fonction calcule tout mais ne crée pas de fichier.
    """
    folder_path = Path(year_folder_name)
    cavity_folders = [path for path in folder_path.iterdir() if path.is_dir()]
    
    files_list = []
    for cavity_folder in cavity_folders:
        if "CMA" in cavity_folder.parts or "CMB" in cavity_folder.parts:
            subfolders = [path for path in cavity_folder.iterdir() if path.is_dir()]
            for sub in subfolders:
                files_list.extend([p for p in sub.iterdir() if p.is_file()])
                
    year_results = _run_parallel(files_list)
    
    year_str = folder_path.parts[-1]
    
    # On ne crée le CSV annuel QUE si le paramètre save_csv est True
    if save_csv:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True) # Sécurité : crée le dossier s'il n'existe pas
        export_filepath = out_dir / f"LLRF_{export_prefix}{year_str}.csv"

        with open(export_filepath, "w", encoding="utf-8") as f:
            f.write("Fichier;Defaut;Label;Détails\n")
            for file_path in sorted(year_results.keys()):
                label = year_results[file_path]
                f.write(f'{file_path.name};{label}\n')

    # On renvoie les résultats et l'année pour pouvoir les utiliser ailleurs
    return year_results, year_str

@chrono
def run_full_analysis(base_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH, years=None, prefix="", combine_all=False):
    """
    Superviseur : Lance l'analyse sur plusieurs années.
    - Si combine_all=False : Crée un CSV par année (comportement classique).
    - Si combine_all=True : Ne crée AUCUN CSV annuel, et pond un CSV global à la fin.
    """
    if years is None:
        years = ["2020", "2021", "2022", "2023", "2024", "2025"]

    all_combined_results = {}

    for year in years:
        folder_path = Path(base_path) / str(year)
        
        results, year_str = classify_year_folder(folder_path, prefix, save_csv=not combine_all, output_path=output_path)

        # Si on veut tout combiner, on ajoute les résultats de cette année dans notre gros dictionnaire
        if combine_all:
            for file_path, label in results.items():
                all_combined_results[file_path] = (label, year_str)

    # À la toute fin, si on est en mode "combiné", on crée notre fichier unique
    if combine_all and all_combined_results:
        combined_filepath = Path(f"{output_path}LLRF_{prefix}.csv")
        
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined_filepath = out_dir / f"LLRF_{prefix}.csv"
        with open(combined_filepath, "w", encoding="utf-8") as f:
            f.write("Fichier;Annee;Defaut;Label;Détails\n")
            
            for file_path in sorted(all_combined_results.keys()):
                label, year_str = all_combined_results[file_path]
                f.write(f'{file_path.name};{year_str};{label}\n')
                
        print(f"Fichier résultat créé avec succès ! -> {combined_filepath.name}")
