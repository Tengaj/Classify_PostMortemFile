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

# ==========================================
# 1. FONCTIONS UTILITAIRES (FEATURES)
# ==========================================

def analyze_oscillation(signal, dt, snr_threshold=15.0, min_amp_physique=0.005):
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
    #print(f'pic_max : {pic_max}, median yf_pos : {snr_threshold * np.mean(yf_pos)}')
    
    # L'amplitude de l'oscillation est-elle physiquement significative ?
    is_physically_significant = pic_max > min_amp_physique
    
    # Il faut que ce soit une belle oscillation ET qu'elle soit assez grosse
    is_oscillating = is_above_noise and is_physically_significant
    
    #print(f"Freq: {freq_dom:.6f} Hz | Amp: {pic_max:.4f} | Bruit spectral: {np.median(yf_pos):.4f}")
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
    # Calcul de la moyenne et de l'écart-type glissants sur TOUT le fichier
    rolling_mean = Ucav_series.rolling(window=window_size).mean()
    rolling_std = Ucav_series.rolling(window=window_size).std()
    
    # Éviter la division par zéro si le signal est parfaitement plat
    rolling_std = rolling_std.replace(0, 1e-6)
    
    # Calcul du Z-Score global
    z_scores = abs((Ucav_series - rolling_mean) / rolling_std)
    
    # On demande directement la position numérique de l'index 0.0
    if 0.0 in Ucav_series.index:
        pos_t_zero = Ucav_series.index.get_loc(0.0)
    else:
        pos_t_zero = int((Ucav_series.size / 2) - 1)
    
    # Définir les bornes (les -9900 et +100 points autour de ce 0)
    start_pos = max(0, pos_t_zero - 9900)
    end_pos = min(len(z_scores), pos_t_zero + 100 + 1)
    
    # Isoler uniquement la zone qui nous intéresse
    z_scores_cible = z_scores.iloc[start_pos:end_pos]
    
    # Trouver le premier point qui dépasse le seuil DANS LA ZONE CIBLE
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

@chrono
def classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, t0_idx, window_points=50):
    """
    Classifie le défaut en appliquant l'arbre de décision.
    - t0_idx : l'index dans les tableaux correspondant au moment du défaut (trip).
    - window_points : le nombre de points à analyser JUSTE AVANT t0.
    """
    # On isole la fenêtre d'analyse "proche de t0" (avant la coupure)
    # A voir si c'est nécessaire    
    #print(f'length time : {time.shape}')
    start_idx = max(0, t0_idx - window_points)
    end_idx = t0_idx + 10
    zero_arg = np.argwhere(time == 0)
    if zero_arg.size == 0:
        zero_idx = int((time.size / 2) - 1)
    else:
        zero_idx = zero_arg[0][0] #Indice ou time = 0, t0_idx est le moment ou Ucav commence à chuter
    print(f't0_idx : {t0_idx}, t0 valeur : {time[t0_idx]}')
    print(f'index de 0 : {zero_idx}')
    print(f'start : {start_idx}, end : {end_idx}')
    
    t_win     = time[start_idx:end_idx]
    Uci_win   = Uci.iloc[start_idx:end_idx]
    Ucr_win   = Ucr.iloc[start_idx:end_idx]
    Ucav_win  = Ucav.iloc[start_idx:end_idx]
    modIQ_win = modIQ.iloc[start_idx:end_idx]

    dt_microseconds = np.mean(np.diff(time))
    dt_seconds = dt_microseconds * 1e-6 # Conversion en secondes pour obtenir la fréquence en Hz et pas Mhz
    osc_Uci, f_Uci = analyze_oscillation(Uci.iloc[t0_idx:zero_idx], dt_seconds)
    osc_modIQ, f_modIQ = analyze_oscillation(modIQ.iloc[t0_idx:zero_idx], dt_seconds)
    osc_Ucav, f_Ucav = analyze_oscillation(Ucav.iloc[t0_idx:zero_idx], dt_seconds)
    
    # Si les trois oscillent (on prend la fréquence de la cavité Ucav comme référence)
    if osc_Uci and osc_modIQ and osc_Ucav:
        freq = f_Ucav 
        if freq > 100:
            return "Loop instability" # A l'air de marcher
        else:
            return "Taconis"
        
    # Analyse des tendances (Pentes)
    #print(f'max modIQ : {modIQ.max()}, pos max modIQ : {modIQ.max()}')
    trend_modIQ = get_trend(t_win, modIQ_win)
    trend_Uci   = get_trend(t_win, Uci_win)
    trend_Ucr   = get_trend(t_win, Ucr_win)
    '''
    #trend_modIQ = get_trend(time, modIQ)
    #trend_Uci   = get_trend(time, Uci)
    #trend_Ucr   = get_trend(time, Ucr)
    print(f'trend moqIQ : {trend_modIQ}, trend Uci : {trend_Uci}, trend Ucr : {trend_Ucr}')

    # Arbre de décision pour modIQ croissant
    if trend_modIQ == 1:
        if trend_Uci == -1:
            return "Power chain gain"
        elif trend_Uci == 1:
            return "Beam loss"
    '''
    # Calcul des vraies pentes
    pente_modIQ = get_slope(t_win, modIQ_win)
    pente_Uci   = get_slope(t_win, Uci_win)
    pente_Ucr   = get_slope(t_win, Ucr_win)
    
    print(f'Pentes brutes -> modIQ: {pente_modIQ:.4f} | Uci: {pente_Uci:.4f} | Ucr: {pente_Ucr:.4f}')

    SEUIL_BRUIT = 0.01        # En dessous de ça (ex: 0.1), on considère que c'est plat
    SEUIL_FORTE_MONTEE = 15.0 # Au-dessus de ça, c'est une montée brutale
    
    # Si modIQ est croissant (et que ce n'est pas juste du bruit)
    if pente_modIQ > SEUIL_BRUIT:
        print(f'pente modIQ : {pente_modIQ}, pente Uci : {pente_Uci}')
        print(f'pente uci < -seuil bruit {pente_Uci < -SEUIL_BRUIT}')
        
        # CAS 1 : Grosse montée de modIQ ET baisse de Uci
        if pente_modIQ > SEUIL_FORTE_MONTEE and pente_Uci < -SEUIL_BRUIT:
            return "Power chain gain"
            
        # CAS 2 : Légère montée de modIQ ET légère montée de Uci
        elif SEUIL_BRUIT < pente_modIQ <= SEUIL_FORTE_MONTEE and pente_Uci > SEUIL_BRUIT:
            return "Beam loss"

    # Arbre de décision pour Uci croissant
    if trend_Uci == 1:
        if trend_Ucr == -1:
            return "Quench"
        elif trend_Ucr == 1:
            return "Quench aside"
            
    # Si aucune condition n'est remplie
    return "Other"

@chrono
def classify(file_name):

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
                return default[0] + ", Pas de chute de Ucav"
            return default[0] + ", "+ classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, t0_idx)
        case _:
            return "Other"

if __name__ == "__main__":
    '''
    #file = '../LLRF/2020/CMA/CMA03/2020 12 06 CMA03_0001' #Power chain gain
    file = '../LLRF/2020/CMA/CMA01/2020 10 22 CMA01_0003'
    print(file)
    label = classify(file)
    print(label)
    '''

    #name_file_export = "resultat/2020_CMA_CMA03_liste_Rég signal RF hors tolérance.txt"
    #name_file_export = "resultat/2024_CMA_CMA08_liste_Rég signal RF hors tolérance.txt"
    name_file_export = "resultat/2020_CMA_CMA01_liste_Rég signal RF hors tolérance.txt"
    default_file = {}

    #path_name = "../LLRF/2020/CMA/CMA03"
    #path_name = "../LLRF/2024/CMA/CMA08"
    path_name = "../LLRF/2020/CMA/CMA01"
    path_file = Path(path_name)
    print(path_file)
    post_mortem_files = sorted([chemin for chemin in path_file.iterdir() if chemin.is_file()])
    for file in post_mortem_files:
        label = classify(file)
        if "Rég signal RF hors tolérance" in label:
            default_file[file] = label
        print(f'Nom de fichier : {file}, Défaut : {label}')

    with open(name_file_export, "w", encoding="utf-8") as f:
        f.write(f'{path_name}\n')
        for key in default_file.keys():
            f.write(f'{key} : {default_file[key]}\n')


#Seuil pick-up,
#Coupure externe rapide,    
#Absence autorisation RF,
#Seuil de vide,
#Claquage ou quench cavité,
#Dép seuil de sécurité RF,
#Rég signal RF hors tolérance