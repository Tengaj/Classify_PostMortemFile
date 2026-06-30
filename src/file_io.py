from pathlib import Path

from utils import chrono, CONFIG
from classification import classify_file

EXEC_CFG = CONFIG.get("execution", {})
DEFAULT_INPUT_PATH = EXEC_CFG.get("input_path", "../../LLRF/")
DEFAULT_OUTPUT_PATH = EXEC_CFG.get("output_path", "../resultat/")

@chrono
def classify_folder(folder_name):
    """
    Classifie tous les fichiers d'un dossiers. 
    ATTENTION : Le path ne doit pas être absolue !!!
    Il doit être du type :  ../LLRF/year/cav_type/cav_number (exemple : ../../LLRF/2024/CMA/CMA09/)
    """
    folder_path = Path(folder_name)
    files_list = sorted([path for path in folder_path.iterdir() if path.is_file()])
    folder_results = {}
    
    for file_path in files_list:
        if file_path.stat().st_size == 0:
            print(f'Fichier ignoré : {file_path} (0 octet sur le disque)')
            continue
        try:
            label = classify_file(file_path)
            if "Rég signal RF hors tolérance" in label or "Dép seuil de sécurité RF" in label or "Seuil de vide" in label or "Claquage ou quench cavité" in label:
            #if "Rég signal RF hors tolérance" in label:
            #if "Seuil de vide" in label:
                folder_results[file_path] = label
            #file_default[file] = label
            print(f'Nom du fichier : {file_path}, Défaut : {label}')

        except Exception as e:
            print(f'Fichier ignoré ({file_path}) - Fichier imcomplet ou erreur : {e}')
            continue

    return folder_results

@chrono
def classify_cav_folder(cavity_folder_name):
    """
    Classifie tous les fichiers des sous dossiers dun type de cavité.
    """
    folder_path = Path(cavity_folder_name)
    subfolders = sorted([path for path in folder_path.iterdir() if path.is_dir()])
    cavity_results = {}
    
    for subfolder in subfolders:
        cavity_results.update(classify_folder(subfolder))
        
    return cavity_results

@chrono
def classify_year_folder(year_folder_name, export_prefix='', save_csv=True, output_path=DEFAULT_OUTPUT_PATH):
    """
    Classifie tous les fichiers des sous-dossiers d'une année.
    Exporte le résultat en CSV si save_csv est True.
    """
    folder_path = Path(year_folder_name)
    cavity_folders = sorted([path for path in folder_path.iterdir() if path.is_dir()])
    year_results = {}
    
    for cavity_folder in cavity_folders:
        print(cavity_folder)
        if "CMA" in cavity_folder.parts or "CMB" in cavity_folder.parts:
            year_results.update(classify_cav_folder(cavity_folder))
            
    year_str = folder_path.parts[-1]

    if save_csv:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_filepath = out_dir / f"LLRF_{export_prefix}{year_str}.csv"
        
        with open(export_filepath, "w", encoding="utf-8") as f:
            f.write("Fichier;Defaut;Label;Détails\n")
            # Tri des résultats pour avoir un CSV propre
            for file_path in sorted(year_results.keys()):
                label = year_results[file_path]
                f.write(f'{file_path.name};{label}\n')
                
    # On retourne les résultats et l'année pour la fonction superviseur
    return year_results, year_str

@chrono
def run_full_analysis(base_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH, years=None, prefix="", combine_all=False):
    """
    Lance l'analyse sur plusieurs années l'une après l'autre.
    - Si combine_all=False : Crée un CSV par année.
    - Si combine_all=True : Ne crée AUCUN CSV annuel, et pond un CSV global à la fin.
    """
    if years is None:
        years = ["2020", "2021", "2022", "2023", "2024", "2025"]

    all_combined_results = {}

    for year in years:
        folder_path = Path(base_path) / str(year)
        print(f"\n{'='*50}")
        print(f" Démarrage de l'analyse (SÉQUENTIELLE) pour l'année {year}")
        print(f"{'='*50}")
        
        # On passe save_csv=False si on veut tout combiner à la fin
        results, year_str = classify_year_folder(folder_path, prefix, save_csv=not combine_all, output_path=output_path)

        if combine_all:
            for file_path, label in results.items():
                all_combined_results[file_path] = (label, year_str)

    # Création du fichier combiné final si demandé
    if combine_all and all_combined_results:
        print("\n--- Création du fichier CSV global unique ---")
        
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined_filepath = out_dir / f"LLRF_{prefix}.csv"
        
        with open(combined_filepath, "w", encoding="utf-8") as f:
            f.write("Fichier;Annee;Defaut;Label;Détails\n")
            
            for file_path in sorted(all_combined_results.keys()):
                label, year_str = all_combined_results[file_path]
                f.write(f'{file_path.name};{year_str};{label}\n')
                
        print(f"Fichier unique créé avec succès ! -> {combined_filepath.name}")