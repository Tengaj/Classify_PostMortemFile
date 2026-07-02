# Classification automatique des défauts dans les cavités supraconductrices RF de SPIRAL 2

Ce projet permet l'analyse post-mortem et la classification automatique des signaux LLRF (Low Level Radio Frequency). Il traite les données brutes pour identifier la nature physique des pannes. Il permet une analyse annuelle ou complète des fichiers post-mortem. 

## Prérequis

Ce programme fonctionne sous Python 3.10 (développé et testé sur la version 3.10.12).

Il nécessite les bibliothèques externes suivantes (non incluses par défaut dans Python) :
 - NumPy
 - Pandas
 - SciPy
 - PyYAML

Pour installer rapidement toutes les dépendances publiques, vous pouvez exécuter la commande suivante dans votre terminal :

```bash
pip install numpy pandas scipy pyyaml
```

## Utilisation et lancement 

Avant toute utilisation, assurez-vous que les chemins d'accès d'entrée et de sortie sont corrects. Ils sont modifiables dans le fichier `config.yaml`, au sein de la section execution. La valeur `input_path` correspond au chemin d'accès vers les fichiers post-mortem, et `output_path` indique le dossier où seront enregistrés les résultats.

Ensuite, placez-vous dans le répertoire source et lancez le script principal :

```Bash

cd src/
python3 main.py

```

## Format des résultats

Les résultats de l'analyse sont exportés sous forme de fichiers CSV (séparateur : point-virgule `;`). Selon votre configuration (`combine_csvs`), vous obtiendrez soit un fichier par année, soit un fichier global. 

Le fichier généré contient les colonnes suivantes :

 - **Fichier** : Le nom exact du fichier de données brutes analysé.
 - **Annee** : L'année de l'événement *(colonne présente uniquement si l'option de fichier combiné global est activée)*.
 - **Defaut** : La catégorie majeure du défaut renseignée par la machine (ex: "Rég signal RF hors tolérance").
 - **Label** : La classification physique précise trouvée par l'algorithme (ex: "Quench", "Taconis", "Other").
 - **Détails** : Informations supplémentaires sur le fichier (ex: temps de chute pour un "Quench" ou la raison de filtrage).

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