from pathlib import Path
from utils import chrono
from classification import classify_file

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
            #if "Rég signal RF hors tolérance" in label or "Dép seuil de sécurité RF" in label or "Seuil de vide" in label or "Claquage ou quench cavité" in label:
            if "Rég signal RF hors tolérance" in label:
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
def classify_year_folder(year_folder_name,export_prefix = ''):
    """
    Classifie tous les fichiers des sous-dossiers d'une année et exporte le résultat en CSV.
    """
    folder_path = Path(year_folder_name)
    cavity_folders = sorted([path for path in folder_path.iterdir() if path.is_dir()])
    year_results = {}
    
    for cavity_folder in cavity_folders:
        print(cavity_folder)
        if "CMA" in cavity_folder.parts or "CMB" in cavity_folder.parts:
            year_results.update(classify_cav_folder(cavity_folder))
            
    year = folder_path.parts[-1]
    export_filepath = f"../resultat/LLRF_{export_prefix}{year}.csv"
    
    with open(export_filepath, "w", encoding="utf-8") as f:
        f.write("Fichier;Defaut;Label;Détails\n")
        
        for file_path, label in year_results.items():
            file_name = file_path.name
            f.write(f'{file_name};{label}\n')