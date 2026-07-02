# Classification automatique des défauts dans les cavités supraconductrices RF de SPIRAL 2

Ce projet permet l'analyse post-mortem et la classification automatique des signaux LLRF (Low Level Radio Frequency). Il traite les données brutes pour identifier la nature physique des pannes.

## Utilisation et lancement 

Avant toute utilisation, assurez-vous que les chemins d'accès d'entrée et de sortie sont corrects. Ils sont modifiables dans le fichier `config.yaml`, au sein de la section execution. La valeur `input_path` correspond au chemin d'accès vers les fichiers post-mortem, et `output_path` indique le dossier où seront enregistrés les résultats.

Ensuite, placez-vous dans le répertoire source et lancez le script principal :

```Bash

cd src/
python3 main.py

```

## Configuration 

Cette classification est entièrement paramétrable via le fichier config.yaml.

 - execution : Vous y trouverez tout ce qui est relatif à l'exécution du programme (chemins d'accès, années à analyser, options de compilation des CSV, etc.).

 - filters : Vous permet d'activer ou de désactiver chaque filtre de validité, ainsi que de modifier leurs valeurs de seuil.

 - thresholds : Liste les seuils physiques principaux relatifs à la classification algorithmique, regroupés par défaut majeur.

## Architecture du code

`classification.py` : Contient le cœur de la logique de classification et les arbres de décision.

`signal_processing.py` : Regroupe toutes les fonctions relatives à l'analyse mathématique directe des signaux.

`file_io.py` et `file_io_para.py` : Gèrent les entrées/sorties (lecture des dossiers et enregistrement des résultats). La variante _para utilise le traitement parallélisé multi-cœurs.

`utils.py` : Méthodes utilitaires.

`PyPostMortem.py` : Gère la lecture brute des fichiers post-mortem pour leur exploitation (Fichier issu des travaux de Charly LASSALLE : [Lien GitLab](https://gitlab.in2p3.fr/charly-lassalle-phd-thesis/programmes)).

## Liste des anomalies classifiés

Voici la liste des anomalies détectées et classifiées par le programme, regroupées par défaut principal :

 - Seuil de vide :
    - Ipu
    - Leak/Outgassing
 - Claquage ou quench cavité :
    - Breakdown
    - False alarm
 - Dép seuil de sécurité RF :
    - Ucav max
    - Pr max
 - Rég signal RF hors tolérance :
    - Loop instability
    - Taconis
    - Quench
    - Quench aside
    - Power chain gain
    - Beam loss
    - Beam anomaly
    - Detuning control error
    - Wrong alarm

Note : Si un fichier ne remplit aucune des conditions pour être classé parmi les catégories ci-dessus, il sera étiqueté comme Other.