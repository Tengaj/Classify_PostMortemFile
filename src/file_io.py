from pathlib import Path
from utils import chrono
from classification import classify_file

@chrono
def classify_folder(folder_name):
    """
    Classifie tous les fichiers d'un dossiers. 
    Enregistre le défaut et le label de chaque fichier dans un .csv
    ATTENTION : Le path ne doit pas être absolue !!!
    Il doit être du type :  ../LLRF/year/cav_type/cav_number (exemple : ../../LLRF/2024/CMA/CMA09/)
    """
    # Récupération de tous les fichiers du dossier
    path_folder = Path(folder_name)
    post_mortem_files = sorted([path for path in path_folder.iterdir() if path.is_file()])
    file_default = {}
    for file in post_mortem_files:
        if file.stat().st_size == 0:
            print(f'Fichier ignoré : {file} (0 octet sur le disque)')
            continue
        try:
            label = classify_file(file)
            #Pour le moment je ne regard que le défaut : Rég signal RF hors tolérance
            #if "Rég signal RF hors tolérance" in label or "Dép seuil de sécurité RF" in label or "Seuil de vide" in label or "Claquage ou quench cavité" in label:
            if "Rég signal RF hors tolérance" in label:
            #if "Seuil de vide" in label:
                file_default[file] = label
            #file_default[file] = label
            print(f'Nom du fichier : {file}, Défaut : {label}')
        except Exception as e:
            print(f'Fichier ignoré ({file}) - Fichier imcomplet ou erreur : {e}')
            continue

    return file_default

    # Ecriture des résultats

    # Création du nom du fichier csv
    path_parts = path_folder.parts
    year = path_parts[-3]
    folder_name_clean = ""
    for part in path_parts:
        if part not in ('..', '.'):
            if part == path_parts[len(path_parts)-1]:
                folder_name_clean += part
            else:
                folder_name_clean += part + "_"

    file_name_export = "../resultat/" + year + "/" + folder_name_clean + ".csv"
    with open(file_name_export, "w", encoding="utf-8") as f:
        f.write("Fichier,Defaut,Label,Détails\n")
        for file in file_default.keys():
            file_name = file.parts[len(file.parts)-1]
            f.write(f'{file_name};{file_default[file]}\n')

@chrono
def classify_cav_folder(cav_folder_name):
    """
    Classifie tous les fichiers des sous dossiers dun type de cavité.
    """
    path_folder = Path(cav_folder_name)
    cav_number_folder = sorted([path for path in path_folder.iterdir() if path.is_dir()])
    file_default = {}
    for folder in cav_number_folder:
        file_default.update(classify_folder(folder))
    return file_default

@chrono
def classify_year_folder(year_folder_name,export = ''):
    """
    Classifie tous les fichiers des sous dossiers d'une année.
    """
    path_folder = Path(year_folder_name)
    cav_folder = sorted([path for path in path_folder.iterdir() if path.is_dir()])
    file_default = {}
    for folder in cav_folder:
        print(folder)
        if "CMA" in folder.parts or "CMB" in folder.parts:
            file_default.update(classify_cav_folder(folder))
    year = path_folder.parts[-1]
    file_name_export = "../resultat/LLRF_" + export + year + ".csv"
    with open(file_name_export, "w", encoding="utf-8") as f:
        f.write("Fichier;Defaut;Label;Détails\n")
        for file in file_default.keys():
            file_name = file.parts[len(file.parts)-1]
            f.write(f'{file_name};{file_default[file]}\n')