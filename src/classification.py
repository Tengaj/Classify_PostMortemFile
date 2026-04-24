import base64
import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
from functools import wraps
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
from scipy.signal import detrend
from PyPostMortem import Read_Signals

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

# FONCTIONS UTILITAIRES

def analyze_oscillation(signal, dt, snr_threshold=5.0, min_amp_physique=0.005):
    """
    Détermine si le signal oscille et renvoie la fréquence
    - snr_threshold : Rapport signal/bruit dans le spectre (relatif).
    - min_amp_physique : Amplitude physique minimale en dessous de laquelle 
                         on considère que c'est juste le bruit de la carte LLRF.
                         (À ajuster selon l'unité de tes données : Volts, MV/m, etc.)
    """
    if len(signal) < 10:    
        return False, 0.0 

    n = len(signal)
    
    # Enlever la pente et la moyenne (Detrend linéaire)
    sig_detrended = detrend(signal, type='linear')
    
    # Fenêtrage (Hanning Window)
    # Adoucit les bords de la fenêtre pour éviter les faux pics de coupure
    window = np.hanning(n)
    sig_windowed = sig_detrended * window
    
    # Calcul de la FFT
    yf = np.abs(fft(sig_windowed))
    xf = fftfreq(n, d=dt)
    
    # Normalisation
    yf = 2.0 / n * yf

    # La résolution fréquentielle (df) est l'écart entre 2 points de la FFT.
    # On ignore les 5 premiers points qui contiennent toujours la fuite du detrend.
    freq_resolution = 1.0 / (n * dt)
    freq_min_valid = 5 * freq_resolution
    
    # On ne garde que les fréquences positives
    mask = xf > freq_min_valid

    # Si la fenêtre est trop petite et qu'il n'y a plus de fréquences valides
    if not np.any(mask):
        return False, 0.0
    xf_pos, yf_pos = xf[mask], yf[mask]
    
    # Recherche du pic dominant
    idx_max = np.argmax(yf_pos)
    freq_dom = xf_pos[idx_max]
    pic_max = yf_pos[idx_max]

    #  Le pic sort-il vraiment du bruit de fond spectral ?
    is_above_noise = pic_max > (snr_threshold * np.median(yf_pos))
    print(f'snr threshold : {snr_threshold}, min amp physique : {min_amp_physique}')
    print(f'pic_max : {pic_max:.5f}, median yf_pos : {snr_threshold * np.mean(yf_pos):.5f}')
    
    # L'amplitude de l'oscillation est-elle physiquement significative ?
    is_physically_significant = pic_max > min_amp_physique
    
    # Il faut que ce soit une belle oscillation ET qu'elle soit assez grosse
    is_oscillating = is_above_noise and is_physically_significant
    
    print(f"Freq: {freq_dom:.6f} Hz | Amp: {pic_max:.4f} | Bruit spectral: {np.median(yf_pos):.4f} | is_oscillating : {is_oscillating}")
    #print(f'is_oscillating : {is_oscillating}')
    
    return is_oscillating, freq_dom

def get_trend(time, signal, tolerance=1e-5):
    """
    Détermine si le signal est croissant, décroissant ou stable
    en utilisant la pente de la régression linéaire.
    Retourne 1 si le signal est croissant, -1 si il est décroissant et 0 sinon
    La tolérance permet d'ignorer le bruit plat.
    """
    slope, intercept, r_value, p_value, std_err = linregress(time, signal)
    #print(f'slope : {slope}')
    
    if slope > tolerance:
        return 1
    elif slope < -tolerance:
        return -1
    else:
        return 0

def detect_t0_zscore(Ucav_series, window_size=10000, z_threshold=5.0):
    """
    Détecte t0, le moment ou le signal Ucav sort de son amplitude moyenne avec un Z-score glissant.
    - Ucav_series : Doit être une pd.Series.
    - window_size : Taille de la fenêtre d'apprentissage du bruit.
    - z_threshold : Nombre d'écarts-types pour déclencher l'alarme (5 est standard).
    """
    # Calcul de la moyenne et de l'écart-type glissants
    rolling_mean = Ucav_series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = Ucav_series.rolling(window=window_size, min_periods=1).std()

    # Éviter la division par zéro si le signal est parfaitement plat
    rolling_std = rolling_std.replace(0, 1e-6)

    '''
    temps_cible = -54319.0
    ligne_exacte = Ucav_series.index.get_indexer([temps_cible], method='nearest')[0]
    vrai_temps = Ucav_series.index[ligne_exacte]
    val_brute = Ucav_series.iloc[ligne_exacte]
    val_mean  = rolling_mean.iloc[ligne_exacte]
    val_std   = rolling_std.iloc[ligne_exacte]
    print(f"Temps cherché: {temps_cible} | Trouvé: {vrai_temps}")
    print(f"Val : {val_brute:.4f} | Mean : {val_mean:.4f} | std : {val_std:.4f}")
    '''

    # Calcul du Z-Score global
    z_scores = abs((Ucav_series - rolling_mean) / rolling_std)
    
    if 0.0 in Ucav_series.index:
        # On demande directement la position numérique de l'index 0.0
        pos_t_zero = Ucav_series.index.get_loc(0.0)
    else:
        # On prend la position central du signal
        pos_t_zero = int((Ucav_series.size / 2) - 1)
    
    # Définir les bornes (les -9900 et +100 points autour de ce 0)
    start_pos = max(0, pos_t_zero - 9900)
    end_pos = min(len(z_scores), pos_t_zero + 100 + 1)
    
    # Isoler uniquement la zone qui nous intéresse
    z_scores_cible = z_scores.iloc[start_pos:end_pos]
    
    # Trouver le premier point qui dépasse le seuil dans la zone cible
    anomalies = z_scores_cible[z_scores_cible > z_threshold]

    if not anomalies.empty:
        # On prend le premier temps où l'anomalie est détectée
        t0_time = anomalies.index[0]
        t0_idx = Ucav_series.index.get_loc(t0_time)
        return t0_idx, t0_time
        
    return None, None

def get_slope(t_array, y_array):
    """
    Calcule la pente du signal sur une fenêtre de temps normalisée.
    La pente renvoyée équivaut à la variation TOTALE de l'amplitude sur la fenêtre.
    """
    # On s'assure d'avoir des tableaux NumPy plats
    # (Au cas où on lui passerait des Series Pandas)
    t_vals = np.array(t_array).flatten()
    y_vals = np.array(y_array).flatten()
    
    # Sécurité : si la fenêtre est vide
    if len(t_vals) < 2:
        return 0.0
        
    # NORMALISATION DU TEMPS (de 0.0 à 1.0)
    t_min = t_vals.min()
    t_max = t_vals.max()
    
    # Sécurité anti division par zéro
    if t_max == t_min:
        return 0.0
        
    t_norm = (t_vals - t_min) / (t_max - t_min)
    
    # Calcul de la pente
    pente = np.polyfit(t_norm, y_vals, 1)[0]
    
    return pente

def classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, t0_idx, window_points=1000):
    """
    Classifie le défaut en appliquant l'arbre de décision.
    - t0_idx : l'index dans les tableaux correspondant au moment du défaut (trip).
    - window_points : le nombre de points à analyser JUSTE AVANT t0.
    """
    if np.max(np.abs(Uci)) < 0.001:
        print(f"Fichier ignoré/classé : Signal Uci inexistant (Max = {np.max(np.abs(Uci)):.4f})")
        return "Uci nulle" 

    # t0_idx : index ou le signal Ucav comment à décrocher
    # zero_idx : index de 0.0 dans time, si pas de 0.0 dans time, index central   
    zero_arg = np.argwhere(time == 0)
    if zero_arg.size == 0:
        zero_idx = int((time.size / 2) - 1)
    else:
        zero_idx = zero_arg[0][0] #Indice ou time = 0, t0_idx est le moment ou Ucav commence à chuter
    print(f't0_idx : {t0_idx}, t0 valeur : {time[t0_idx]}, index de 0 : {zero_idx}')
    #print(f'start : {start_idx}, end : {end_idx}')

    SEUILS_AMPLITUDE = {
        "Uci":  0.001,
        "modIQ": 0.1,
        "Ucav": 0.0001
    }

    dt_microseconds = np.mean(np.diff(time))
    dt_seconds = dt_microseconds * 1e-6 # Conversion en secondes pour obtenir la fréquence en Hz et pas Mhz

    osc_Uci, f_Uci = analyze_oscillation(
        Uci.iloc[t0_idx:zero_idx+30], 
        dt_seconds, 
        min_amp_physique=SEUILS_AMPLITUDE["Uci"]
    )
    
    osc_modIQ, f_modIQ = analyze_oscillation(
        modIQ.iloc[t0_idx:zero_idx+30], 
        dt_seconds, 
        min_amp_physique=SEUILS_AMPLITUDE["modIQ"]
    )
    
    osc_Ucav, f_Ucav = analyze_oscillation(
        Ucav.iloc[t0_idx:zero_idx+30], 
        dt_seconds,  
        min_amp_physique=SEUILS_AMPLITUDE["Ucav"]
    )
    
    # Si les trois oscillent (on prend la fréquence de la cavité Ucav comme référence)
    if osc_Uci and osc_modIQ and osc_Ucav:
        freq = f_Ucav 
        if freq > 100:
            return "Loop instability" # A l'air de marcher
        else:
            return "Taconis"
    
    # Calcul du "Niveau de repos" (Baseline)
    # On prend une fenêtre "saine" bien avant t0 (ex: entre t0-500 et t0-100)
    idx_base_start = max(0, t0_idx - 500)
    idx_base_end = max(1, t0_idx - 100)
    base_modIQ = np.median(modIQ.iloc[idx_base_start:idx_base_end])
    base_Uci   = np.median(Uci.iloc[idx_base_start:idx_base_end])
    base_Ucr   = np.median(Ucr.iloc[idx_base_start:idx_base_end])
    #print(f'median modIQ : {base_modIQ}, median Uci : {base_Uci}, median Ucr : {base_Ucr}')
    
    # Définition de la Zone de Recherche de Pic (Autour de t0)
    # On autorise le code à chercher un peu avant ET un peu après t0
    # On limite la fin à zero_idx pour ne pas analyser la chute libre finale
    zone_start = max(0, min(t0_idx, zero_idx))
    #zone_end   = min(zero_idx, t0_idx + 50) 
    zone_end   = zero_idx + 20
    
    # Extraction des comportements extrêmes dans cette zone
    max_modIQ_zone = modIQ.iloc[zone_start:zone_end].max()
    max_Ucr_zone = Ucr.iloc[zone_start:zone_end].max()
    max_Uci_zone = Uci.iloc[zone_start:zone_end].max()
    min_Uci_zone = Uci.iloc[zone_start:zone_end].min()
    
    # Pour Uci, comme on cherche à savoir s'il a chuté ou monté, 
    # on regarde sa valeur moyenne juste au moment de l'événement
    '''
    eval_end = min(zero_idx, t0_idx + 20)
    if zero_idx <= t0_idx:
        Uci_event = np.mean(Uci.iloc[t0_idx:t0_idx+20])
    else :
        Uci_event = np.mean(Uci.iloc[t0_idx:eval_end])
    #print(f'Uci event : {Uci_event}, eval end : {eval_end}')
    '''
    
    # Calcul des variations réelles (L'événement par rapport au repos)
    delta_modIQ = max_modIQ_zone - base_modIQ
    delta_pos_Uci   = max_Uci_zone - base_Uci
    delta_neg_Uci   = min_Uci_zone - base_Uci
    delta_Ucr   = max_Ucr_zone - base_Ucr
    #print(f'Variations (Deltas) -> modIQ: {delta_modIQ:.4f} | Uci: {delta_Uci:.4f} | Ucr: {delta_Ucr:.4f}')

    # Calcul des variations en POURCENTAGE
    # Sécurité anti-division par (presque) zéro : 
    EPSILON = 0.01
    
    base_modIQ_safe = max(abs(base_modIQ), EPSILON)
    base_Uci_safe = max(abs(base_Uci), EPSILON)
    base_Ucr_safe = max(abs(base_Ucr), EPSILON)
    
    pct_modIQ = (delta_modIQ / base_modIQ_safe) * 100.0
    pct_pos_Uci   = (delta_pos_Uci / base_Uci_safe) * 100.0
    pct_neg_Uci   = (delta_neg_Uci / base_Uci_safe) * 100.0
    pct_Ucr   = (delta_Ucr / base_Ucr_safe) * 100.0
    
    print(f'index start : {zone_start} | index end : {zone_end}, start time value : {time[zone_start]} | end time value : {time[zone_end]}')
    print(f'Deltas absolus -> modIQ: {delta_modIQ:.4f} | Uci_pos: {delta_pos_Uci:.4f} | Uci_neg: {delta_neg_Uci:.4f} | Ucr: {delta_Ucr:.4f}')
    print(f'Variations (%) -> modIQ: {pct_modIQ:.1f}% | Uci_pos: {pct_pos_Uci:.1f}% | Uci_neg: {pct_neg_Uci:.4f}| Ucr: {pct_Ucr:.1f}% ')
    
    SEUIL_BRUIT_PCT = 5.0
    SEUIL_GROS_PIC_PCT = 150.00
    
    if pct_pos_Uci > SEUIL_BRUIT_PCT:
        pct_Uci = pct_pos_Uci
    else:
        pct_Uci = pct_neg_Uci
    
    # Si modIQ monte plus que le bruit ambiant
    if pct_modIQ > SEUIL_BRUIT_PCT:
        
        # CAS 1 : Power Chain Gain (ÉNORME pic > 100% ET Uci baisse d'au moins 5%)
        if pct_modIQ > SEUIL_GROS_PIC_PCT and pct_Uci < -SEUIL_BRUIT_PCT:
            return "Power chain gain"
            
        # CAS 2 : Beam Loss (Montée de modIQ entre 5% et 100% ET montée de Uci > 5%)
        elif SEUIL_BRUIT_PCT < pct_modIQ and pct_Uci > SEUIL_BRUIT_PCT:
            return "Beam loss"

    # On isole la fenêtre d'analyse "proche de t0" (avant la coupure)
    # A voir si c'est nécessaire
    start_idx = max(0, t0_idx - window_points)
    end_idx = t0_idx

    t_win     = time[start_idx:end_idx]
    Uci_win   = Uci.iloc[start_idx:end_idx]
    Ucr_win   = Ucr.iloc[start_idx:end_idx]
    Ucav_win  = Ucav.iloc[start_idx:end_idx]
    modIQ_win = modIQ.iloc[start_idx:end_idx]

    # Analyse des tendances (Pentes)
    trend_modIQ = get_trend(t_win, modIQ_win)
    trend_Uci   = get_trend(t_win, Uci_win)
    trend_Ucr   = get_trend(t_win, Ucr_win)
    print(f'trend moqIQ : {trend_modIQ}, trend Uci : {trend_Uci}, trend Ucr : {trend_Ucr}') 

    # Arbre de décision pour Uci croissant
    if trend_Uci == 1:
        if trend_Ucr == -1:
            return "Quench"
        elif trend_Ucr == 1:
            return "Quench aside"

    if pct_Ucr > SEUIL_GROS_PIC_PCT:
        # Cas particulier pour le moment non répertorié ou Ucr croit seul après t0 tandis que Ucav, Uci et modIQ baisse.
        return "Ucr croissant seul"
            
    # Si aucune condition n'est remplie
    return "Other"

@chrono
def classify_file(file_name):

    t_start_read = perf_counter()

    with open(file_name, 'rb') as f:
        file_content = f.read()

    parameters, time, df_signals, df_defaults, _, header = Read_Signals(
        file_content, 
        compute_defauts=True, 
        compute_etats=False, 
        plot_signaux=False, 
        show_header=False
    )

    t_end_read = perf_counter()
    temps_lecture = t_end_read - t_start_read

    # Création d'un DataFrame de True et de False, True quand un signal est à 0, donc qu'il y a un défaut
    row_default = (df_defaults == 0).any(axis=1)
    # Si il y a un signal à 0
    if row_default.any():
        #Trouve l'index du premier True
        first_time_default = row_default.idxmax()
        #Récupère juste la colonne ou le défaut à lieu
        value_on_time = df_defaults.loc[first_time_default]
        #Récupère les nom des défauts (Si il y en a plusieurs)
        default = value_on_time[value_on_time==0].index.tolist()
    #print(f'Nom du défauf : {default}')
    else:
        default = [""]

    match default[0]:
        case "Seuil pick-up":
            return "1"
        case "Coupure externe rapide":
            return "2"
        case "Absence autorisation RF":
            return "3"
        case "Seuil de vide":
            pickup = df_signals.get("courant pickup")
            vact = parameters.get("VACT")
            return "4"
        case "Claquage ou quench cavité":
            Ucr = df_signals.get("A Ucr") 
            Ucav = df_signals.get("Ucav") 
            return "5"
        case "Deép seuil de sécurité RF":
            return "6"
            Ucr = df_signals.get("A Ucr") 
            Ucav = df_signals.get("Ucav") 
        case "Rég signal RF hors tolérance":
            Uci = df_signals.get("Uci")
            Ucr = df_signals.get("A Ucr") 
            Ucav = df_signals.get("Ucav") 
            modIQ = df_signals.get("IQ")
            t0_idx, _ = detect_t0_zscore(Ucav)
            if t0_idx is None:
                # Si on ne trouve pas de t0, ça sert à rien de lancer la classification
                return default[0] + ",Pas de chute de Ucav"
            return default[0] + ","+ classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, t0_idx)
        case _:
            return "Other"

def classify_folder(folder_name):
    """
    Classifie tous les fichiers d'un dossiers. 
    Enregistre le défaut et le label de chaque fichier dans un .csv
    """

    # Récupération de tous les fichiers du dossier
    path_folder = Path(folder_name)
    post_mortem_files = sorted([path for path in path_folder.iterdir() if path.is_file()])
    file_default = {}
    for file in post_mortem_files:
        try:
            label = classify_file(file)
            #Pour le moment je ne regard que le défaut : Rég signal RF hors tolérance
            if "Rég signal RF hors tolérance" in label:
                file_default[file] = label
            #file_default[file] = label
            print(f'Nom du fichier : {file}, Défaut : {label}')
        except Exception as e:
            print(f'Fichier ignoré ({file}) - Fichier imcomplet ou erreur : {e}')
            continue

    # Ecriture des résultats
    path_parts = path_folder.parts
    folder_name_clean = ""
    for parts in path_parts:
        if parts not in ('..', '.'):
            if parts == path_parts[len(path_parts)-1]:
                folder_name_clean += parts
            else:
                folder_name_clean += parts + "_"

    file_name_export = "resultat/" + folder_name_clean + ".csv"
    with open(file_name_export, "w", encoding="utf-8") as f:
        f.write("Fichier,Defaut,Label\n")
        for file in file_default.keys():
            file_name = file.parts[len(file.parts)-1]
            f.write(f'{file_name},{file_default[file]}\n')

def classify_cav_folder(cav_folder_name):
    """
    Classifie tous les fichiers des sous dossiers dun type de cavité.
    """
    path_folder = Path(cav_folder_name)
    cav_number_folder = sorted([path for path in path_folder.iterdir() if path.is_dir()])
    for folder in cav_number_folder:
        classify_folder(folder)

def classify_year_folder(year_folder_name):
    """
    Classifie tous les fichiers des sous dossiers d'une année.
    """
    path_folder = Path(year_folder_name)
    cav_folder = sorted([path for path in path_folder.iterdir() if path.is_dir()])
    for folder in cav_folder:
        classify_cav_folder(folder)


if __name__ == "__main__":
    '''
    file = '../../LLRF/2020/CMB/CMB06B/2020 11 30 CMB06-CAV2_0009 13h18 50ms' # Détecte bien quench
    print(file)
    label = classify_file(file)
    print(label)
    file = '../../LLRF/2020/CMB/CMB06A/2020 11 30 CMB06-CAV1_0002 quench sur cav à coté' # Détecte bien quench aside
    print(file)
    label = classify_file(file)
    print(label)
    '''

    file = '../../LLRF/2024/CMA/CMA03/2024 11 08 CMA03_0006' #power chain gain
    label = classify_file(file)
    print(f'Nom du fichier : {file} | Label :{label}')
    
    #classify_folder("../../LLRF/2024/CMA/CMA03")
    #classify_cav_folder("../../LLRF/2024/CMA")
    #classify_year_folder("../../LLRF/2024/")

#Seuil pick-up,
#Coupure externe rapide,    
#Absence autorisation RF,
#Seuil de vide,
#Claquage ou quench cavité,
#Dép seuil de sécurité RF,
#Rég signal RF hors tolérance