# Classification automatique des défauts dans les cavités supraconductrices RF de SPIRAL 2

Ce projet permet l'analyse post-mortem et la classification automatique des signaux LLRF (Low Level Radio Frequency) du GANIL. Il traite les données brutes pour identifier la nature physique des pannes. Il permet une analyse annuelle ou complète des fichiers post-mortem. 

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

Pour garantir une reproductibilité totale, nous vous recommandons d'installer les versions exactes des bibliothèques utilisées lors du développement à l'aide du fichier requirements.txt :

```bash
pip install -r requirements.txt
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
 - **Label** : La classification physique précise trouvée par l'algorithme (ex: "Quench", "Beam loss", "Other").
 - **Détails** : Informations supplémentaires sur le fichier (ex: temps de chute pour un "Quench" ou la raison de filtrage).

## Configuration 

Cette classification est entièrement paramétrable via le fichier config.yaml.

 - **execution** : Vous y trouverez tout ce qui est relatif à l'exécution du programme (chemins d'accès, années à analyser, défauts à classifier, options de compilation des CSV, etc.).

 - **filters** : Vous permet d'activer ou de désactiver chaque filtre de validité, ainsi que de modifier leurs valeurs de seuil.

 - **thresholds** : Liste les seuils physiques principaux relatifs à la classification algorithmique, regroupés par défaut majeur.

## Architecture du code

`classification.py` : Contient le cœur de la logique de classification et les arbres de décision.

`signal_processing.py` : Regroupe toutes les fonctions relatives à l'analyse mathématique directe des signaux.

`file_io.py` : Gère les entrées/sorties (lecture des dossiers et enregistrement des résultats), avec une bascule automatique entre un traitement séquentiel ou un traitement parallélisé multi-cœurs.

`utils.py` : Méthodes utilitaires.

`PyPostMortem.py` : Gère la lecture brute des fichiers post-mortem pour leur exploitation (Fichier issu des travaux de Charly LASSALLE : [Lien GitLab](https://gitlab.in2p3.fr/charly-lassalle-phd-thesis/programmes)).

## Liste des anomalies classifiées

Voici la liste des anomalies détectées et classifiées par le programme, regroupées par défaut principal :

 - Claquage ou quench cavité :
    - Claquage
    - Fausse alarme
 - Dép seuil de sécurité RF :
    - Ucav max
    - Pr max
 - Rég signal RF hors tolérance :
    - Oscillation Ucav
    - Quench
    - Quench adjacent
    - Gain chaîne de puissance
    - Perte faisceau
    - Anomalie liée au faisceau
    - Erreur de contrôle d'accord
    - Fausse alerte

*Note : Si un fichier ne remplit aucune des conditions pour être classé parmi les catégories ci-dessus, il sera étiqueté comme **Autre**.*

**Autres défauts pris en charge :**
Si vous activez la classification pour les défauts **Seuil pick-up**, **Coupure externe rapide**, **Absence autorisation RF** et **Seuil de vide**, sachez que ceux-ci ne possèdent pas de sous-catégories. 
- Si le fichier ne passe pas les conditions de validité, le seul label attribué sera **Filtré**. 
- S'il est valide, la classification conservera simplement le nom du défaut principal. 
- **Exception pour le Seuil de vide :** Des métriques physiques spécifiques (courant pickup moyen et valeur du vide) sont calculées par l'algorithme et reportées directement dans la colonne *Détails* du fichier de résultats.