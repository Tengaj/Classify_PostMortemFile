import numpy as np
import pandas as pd
from time import perf_counter
from pathlib import Path
from functools import wraps
from scipy.fft import fft, fftfreq
from scipy.stats import linregress
from scipy.signal import detrend, find_peaks
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

def analyze_oscillation(signal, dt, snr_threshold=5.0, min_physical_amp=0.005):
    """
    Détermine si le signal oscille et renvoie la fréquence
    - snr_threshold : Rapport signal/bruit dans le spectre (relatif).
    - min_physical_amp : Amplitude physique minimale en dessous de laquelle on considère que c'est juste du bruit.
    """
    if len(signal) < 10:    
        return False, 0.0 

    n = len(signal)

    # On sécurise le signal en Numpy array
    sig_array = np.asarray(signal)
    
    # Fenêtre glissante dynamique (20% de la durée de l'emballement fourni)
    # Cela permet à la moyenne de grimper la pente sans l'effacer, 
    # et de s'adapter si ton emballement fait 5ms ou 50ms.
    dyn_window = max(5, int(n * 0.20))
    
    # Le rolling mean (moyenne) est meilleur que le median pour lisser une pente
    baseline = pd.Series(sig_array).rolling(window=dyn_window, center=True, min_periods=1).median().to_numpy()
    
    # En soustrayant la baseline, la "montée" disparaît, le signal devient horizontal.
    sig_detrended = sig_array - baseline

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
    min_valid_freq = 0.5 * freq_resolution
    # On ne garde que les fréquences positives
    mask = xf > min_valid_freq

    # Si la fenêtre est trop petite et qu'il n'y a plus de fréquences valides 
    if not np.any(mask):
        return False, 0.0

    xf_pos, yf_pos = xf[mask], yf[mask]

    # Recherche du pic dominant 
    peaks_indices, _ = find_peaks(yf_pos)
    if len(peaks_indices) > 0:
        # S'il y a des vrais pics trouvés, on sélectionne le plus haut parmi eux
        best_peak_idx = np.argmax(yf_pos[peaks_indices])
        max_idx = peaks_indices[best_peak_idx]
    else:
        # Sécurité : Si aucun pic n'est trouvé (signal plat ou pente pure), on retombe sur argmax
        max_idx = np.argmax(yf_pos)
    dominant_freq = xf_pos[max_idx]
    peak_amp = yf_pos[max_idx]

    noise_level = snr_threshold * np.median(yf_pos)
    
    # Le pic sort-il vraiment du bruit de fond spectral ?
    is_above_noise = peak_amp > noise_level
    
    # L'amplitude de l'oscillation est-elle physiquement significative ?
    is_physically_significant = peak_amp > min_physical_amp

    # Zero-Crossing
    # On calcule la période exacte de l'oscillation (en nombre de points)
    period_points = int(1.0 / (dominant_freq * dt))
    
    # On crée une fenêtre glissante qui fait exactement 3 périodes de large.
    zcross_window_size = max(5, 3 * period_points)
    
    # On recalcule un signal spécifiquement centré pour le Z-crossing
    zcross_baseline = pd.Series(signal).rolling(window=zcross_window_size, center=True, min_periods=1).mean().to_numpy()
    sig_zcross = signal - zcross_baseline

    hysteresis_threshold = 0.3 * peak_amp  
    
    states = np.zeros_like(sig_zcross)
    states[sig_zcross > hysteresis_threshold] = 1
    states[sig_zcross < -hysteresis_threshold] = -1
    
    valid_states = states[states != 0]
    
    if len(valid_states) > 0:
        actual_crossings = np.sum(np.abs(np.diff(valid_states)) == 2)
    else:
        actual_crossings = 0
    
    expected_crossings = 2 * (dominant_freq * (n * dt))
    
    is_crossing_valid = (actual_crossings >= 2) and (actual_crossings >= 0.45 * expected_crossings)
    # Il faut que les 3 conditions soient réunies

    is_oscillating = is_above_noise and is_physically_significant and is_crossing_valid

    # Affichage
    #print(f'snr threshold : {snr_threshold}, min physical amp : {min_physical_amp}')
    #print(f'peak_amp : {peak_amp:.5f}, noise_level : {noise_level:.5f}')
    #print(f'is_above_noise : {is_above_noise} | is_physically_significant : {is_physically_significant} | is_crossing_valid : {is_crossing_valid}')
    #print(f"{signal.name} Freq: {dominant_freq:.0f}Hz | Amp: {peak_amp:.5f} | Spectral noise: {noise_level:.5f} | Z-Cross: {actual_crossings}/{int(expected_crossings)} | is_oscillating: {is_oscillating}")

    return is_oscillating, dominant_freq

def get_trend(time, signal, tolerance=1e-5):
    """
    Détermine si le signal est croissant, décroissant ou stable
    en utilisant la pente de la régression linéaire.
    Retourne 1 si le signal est croissant, -1 si il est décroissant et 0 sinon
    La tolérance permet d'ignorer le bruit plat.
    """
    time_arr = np.asarray(time)
    signal_arr = np.asarray(signal)
    # Protection contre les fenêtres trop petites
    if len(signal_arr) < 2 or len(time_arr) < 2:
        #print(f"{signal.name} | array too small (len={len(signal)})")
        return 0

    slope, intercept, r_value, p_value, std_err = linregress(time, signal)
    if slope > tolerance:
        trend = 1
    elif slope < -tolerance:
        trend =  -1
    else:
        trend = 0
    #print(f'{signal.name} | slope : {slope} | trend : {trend}')
    return trend

def detect_t0_zscore(Ucav_series, z_threshold=5.0):
    """
    Détecte t0, le moment ou le signal Ucav sort de son amplitude moyenne avec un Z-score,
    en ignorant les transitoires pour se concentrer sur la dernière anomalie menant au trip.
    - Ucav_series : Doit être une pd.Series.
    - z_threshold : Nombre d'écarts-types pour déclencher l'alarme (5 est standard).
    """
    zero_idx = np.argmin(np.abs(Ucav_series.index))
    
    Ucav_mean = Ucav_series.rolling(window=1000, min_periods=1).mean() #500

    # On définit la zone sûre
    end_safe_zone = max(100, zero_idx - 20000) # - 0
    safe_zone = Ucav_series.iloc[0:end_safe_zone]
    search_window = min(10000, len(safe_zone))
    # On cherche l'écart type minimum (écart type d'un signal stable)
    all_stds = safe_zone.rolling(window=search_window, min_periods=int(search_window*0.5)).std()
    Ucav_std = max(all_stds.min(), 1e-6) # protection
    #Ucav_std = 0.00053
    #print(f'ucav std : {Ucav_std}')
    
    z_scores = abs((Ucav_series - Ucav_mean) / Ucav_std)
    
    z_scores_amp = abs((Ucav_series - Ucav_mean) / Ucav_std)

    Ucav_diff = Ucav_series.diff(periods=20).fillna(0)
    safe_zone_diff = Ucav_diff.iloc[0:end_safe_zone]
    all_stds_diff = safe_zone_diff.rolling(window=search_window, min_periods=int(search_window*0.5)).std()
    all_means_diff = safe_zone_diff.rolling(window=search_window, min_periods=int(search_window*0.5)).mean()
    
    best_idx_diff = all_stds_diff.idxmin()
    diff_std = max(all_stds_diff[best_idx_diff], 1e-6)
    diff_mean = all_means_diff[best_idx_diff]
    
    z_scores_diff = abs((Ucav_diff - diff_mean) / diff_std)

    #print(f'diff std : {diff_std}')

    # Pour chaque point, on retient la pire des deux anomalies.
    z_scores = pd.DataFrame({'amp': z_scores_amp, 'diff': z_scores_diff}).max(axis=1)
    
    '''
    temps_cible = -23400
    ligne_exacte = Ucav_series.index.get_indexer([temps_cible], method='nearest')[0]
    vrai_temps = Ucav_series.index[ligne_exacte]                            
    val_brute = Ucav_series.iloc[ligne_exacte]
    val_mean  = Ucav_mean.iloc[ligne_exacte]                                                                    
    #val_std   = Ucav_std.iloc[ligne_exacte]
    print(f"Temps cherché: {temps_cible} | Trouvé: {vrai_temps}")
    print(f"Val : {val_brute:.4f} | Mean : {val_mean:.4f} | std : {Ucav_std:.4f}")
    print(f'z score : {abs((val_brute - val_mean) / Ucav_std)}')
    '''
    # Définition des bornes
    z_scores_cible = z_scores.iloc[0:zero_idx + 40]
    
    # On définit combien de points anormaux consécutifs on exige 
    MIN_CONSECUTIVE = 3 
    
    # Masque binaire : 1 si on dépasse le seuil, 0 sinon
    is_anomaly_bin = (z_scores_cible > z_threshold).astype(int)
    
    # On inverse la série pour regarder se qui vient après
    # min() vaudra 1 uniquement si le point actuel ET les suivants sont à 1.
    future_window = is_anomaly_bin.iloc[::-1].rolling(window=MIN_CONSECUTIVE).min().iloc[::-1]
    
    # Le premier point d'une vraie chute vaudra exactement 1.
    valid_anomaly_starts = (future_window == 1)
    
    # On extrait uniquement les Z-scores qui démarrent une vraie chute
    anomalies = z_scores_cible[valid_anomaly_starts]
    if anomalies.empty:
        return zero_idx-10000, None

    Ucav_median = safe_zone.median()
    # On définit un seuil de "survie" (ex: il faut qu'il reste au moins 50% de la tension)
    SEUIL_SURVIE = Ucav_median * 0.5 
    Ucav_cible = Ucav_series.iloc[0:zero_idx + 40]
    
    # Un point est sain SEULEMENT SI le Z-score est bon ET que la cavité est allumée
    is_zscore_ok = (z_scores_cible <= z_threshold)
    is_amplitude_ok = (np.abs(Ucav_cible) > SEUIL_SURVIE)
    is_healthy = (is_zscore_ok & is_amplitude_ok).astype(int)
    
    # Un "Vrai Calme" c'est quand le signal est sain sur au moins 200 points consécutifs.
    true_calm = is_healthy.rolling(window=200, min_periods=1).min() == 1.0
    last_calm_times = true_calm[true_calm == True]
    
    if not last_calm_times.empty:
        # On a trouvé au moins une vraie période de paix. 
        # On prend la fin de cette TOUTE DERNIÈRE période de paix.
        end_of_last_calm = last_calm_times.index[-1]
        
        # Le vrai t0 est la première anomalie qui arrive APRÈS cette ultime paix
        anomalies_finales = anomalies[anomalies.index > end_of_last_calm]
        
        if not anomalies_finales.empty:
            t0_time = anomalies_finales.index[0]
            t0_idx = Ucav_series.index.get_loc(t0_time)
            return t0_idx, t0_time

    # Cas ou le signal n'est jamais calme (ou est mort trop tôt)
    t0_time = anomalies.index[0]
    t0_idx = Ucav_series.index.get_loc(t0_time)
    return t0_idx, t0_time

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

def classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ):
    """
    Classifie le défaut "Rég signal RF hors tolérance". 
    """
    # zero_idx : index de 0.0 dans time (ou la valeur la plus proche de 0.0)
    # Récupération de l'index de la valeur de temps la plus proche de 0.0
    zero_idx = np.argmin(np.abs(time))

    dt_microseconds = np.mean(np.diff(time))
    ucav_threshold = 0.1
    min_duration_us = 200 
    required_points = max(2, int(min_duration_us / dt_microseconds))
    ucav_before_t0 = Ucav.iloc[:zero_idx]
    drop_mask = ucav_before_t0 < ucav_threshold
    if drop_mask.rolling(window=required_points).sum().max() >= required_points:
        return f"Other;Tension inférieur à {ucav_threshold} avant t0"
    
    # t0_idx : index ou le signal Ucav comment à décrocher
    t0_idx, _ = detect_t0_zscore(Ucav)
    if t0_idx is None:
        # Si on ne trouve pas de t0, ça sert à rien de lancer la classification
        return "Other;Pas de chute de Ucav"
    if t0_idx == 0:
        return "Other;Problème t0"

    print(f't0_idx : {t0_idx}, t0 valeur : {time[t0_idx]}, index de 0 : {zero_idx}')
    t0_idx_Ucr = detect_t0_zscore(Uci)

    # On isole la fenêtre d'analyse proche de t0 (avant la coupure)
    # Plage d'analyze pour la tendance global des signaux, et pour pour la tendance plus local
    WINDOW_POINTS_LARGE = 3500
    WINDOW_POINTS_SHORT = 700

    start_idx_large = max(0, t0_idx - WINDOW_POINTS_LARGE)
    start_idx_short = max(0, t0_idx - WINDOW_POINTS_SHORT)
    start_idx_short = t0_idx_Ucr[0]
    end_idx = t0_idx

    t_win     = time[start_idx_short:end_idx]
    #Ucav_win  = Ucav.iloc[start_idx_short:end_idx]
    #modIQ_win = modIQ.iloc[start_idx_short:end_idx]
    t_win_large = time[start_idx_large:end_idx]
    Uci_win   = Uci.iloc[start_idx_large:end_idx]
    Ucr_win   = Ucr.iloc[start_idx_large:end_idx]

    if not start_idx_short >= end_idx:
        # Couper la fenêtre Ucav_win en deux et regarder la trend des deux parties
        # Le milieu et la valeur minimal dans l'intervalle
        start_idx_short = max(0,start_idx_short - 200)
        mid_window = start_idx_short + np.argmin(Ucr.iloc[start_idx_short:end_idx]) 
        if mid_window == start_idx_short:
            mid_window = end_idx - int(WINDOW_POINTS_SHORT * 0.5)
        #print(f'Window start : {start_idx_short}, Window mid : {mid_window}, Window end : {end_idx}')
        #print(f'Window start value : {time[start_idx_short]}, Window mid value: {time[mid_window]}, Window end value: {time[end_idx]}')

        t_win_left = time[start_idx_short:mid_window]
        t_win_right = time[mid_window:end_idx]
        Ucr_win_left = Ucr.iloc[start_idx_short:mid_window]
        Ucr_win_right = Ucr.iloc[mid_window:end_idx]

        # Analyse des tendances
        #trend_modIQ = get_trend(t_win, modIQ_win)
        trend_Uci   = get_trend(t_win_large, Uci_win)
        trend_Ucr   = get_trend(t_win_large, Ucr_win)
        trend_Ucr_left = get_trend(t_win_left,Ucr_win_left,tolerance=1e-6)
        trend_Ucr_right = get_trend(t_win_right,Ucr_win_right,tolerance=1e-6)
        #print(f'trend moqIQ : {trend_modIQ}, trend Uci : {trend_Uci}, trend Ucr : {trend_Ucr}, trend Ucr left : {trend_Ucr_left}, trend Ucr right : {trend_Ucr_right} ')
        #print(f'trend Uci : {trend_Uci}, trend Ucr : {trend_Ucr}, trend Ucr left : {trend_Ucr_left}, trend Ucr right : {trend_Ucr_right} ')

        # On calcule la vraie perte d'amplitude entre le début et le point le plus bas
        valeur_debut_Ucr = Ucr.iloc[start_idx_short]
        valeur_min_Ucr = Ucr.iloc[mid_window]
        fall_depth_Ucr = valeur_debut_Ucr - valeur_min_Ucr
        #print(f'fall_depth_Ucr : {fall_depth_Ucr}')
        
        # Seuil en dessous du quel on considère que se n'est pas un quench
        THRESHOLD_QUENCH = 0.1 
        
        # On définit ce qu'est un vrai profil en V pour un quench
        true_V_profil = (trend_Ucr_left == -1 and trend_Ucr_right == 1 and fall_depth_Ucr > THRESHOLD_QUENCH)

        if trend_Uci == 1:
            # Si on a un vrai creux profond, OU une descente globale nette
            if true_V_profil or trend_Ucr == -1:
                return "Quench"
                
            # Si la tendance est à la hausse
            elif trend_Ucr == 1:
                return "Quench aside"

    dt_seconds = dt_microseconds * 1e-6 # Conversion en secondes pour obtenir la fréquence en Hz et pas Mhz 

    SEUILS_AMPLITUDE = {
        "Uci":  0.001,
        "modIQ": 0.1,
        "Ucav": 0.001
    }
    
    if t0_idx > zero_idx:
        start_osci = max(0, zero_idx - 2000)
        end_osci = zero_idx
    else:
        start_osci = t0_idx
        end_osci = zero_idx + 10

    osc_Uci, f_Uci = analyze_oscillation(
        Uci.iloc[start_osci:end_osci], 
        dt_seconds, 
        min_physical_amp=SEUILS_AMPLITUDE["Uci"]
    )
    
    osc_modIQ, f_modIQ = analyze_oscillation(
        modIQ.iloc[start_osci:end_osci], 
        dt_seconds, 
        min_physical_amp=SEUILS_AMPLITUDE["modIQ"]
    )
    
    osc_Ucav, f_Ucav = analyze_oscillation(
        Ucav.iloc[start_osci:end_osci], 
        dt_seconds,  
        min_physical_amp=SEUILS_AMPLITUDE["Ucav"]
    )
    
    # Si les trois oscillent
    if osc_Uci and osc_modIQ and osc_Ucav:
        
        frequencies = [f_Uci, f_modIQ, f_Ucav]
        f_max = max(frequencies)
        f_min = max(min(frequencies), 1e-6)
        
        ratio = f_max / f_min
        #print(f'Ratio : {ratio}')
        closest_integer = round(ratio)
        
        is_coherent = False
        
        # Fondamentale (les fréquences sont quasi-identiques)
        if ratio <= 1.5:
            is_coherent = True
            
        # Harmonique (Le ratio est très proche d'un entier, de la 2ème à la 5ème harmonique)
        elif 2 <= closest_integer <= 5:
            # tolérance fixe
            if abs(ratio - closest_integer) <= 0.2:
                is_coherent = True
        
        # Classification finale si la cohérence est validée
        if is_coherent:
            freq = f_Ucav 
            if freq > 100:
                return "Loop instability"
            else:
                return "Taconis"
        else:
            return "Oscillation"
    
    # Calcul du "Niveau de repos" (Baseline)
    # On prend une fenêtre "saine" bien avant t0 (ex: entre t0-500 et t0-100)
    idx_base_start = max(0, t0_idx - 500)
    idx_base_end = max(1, t0_idx - 100)
    base_modIQ = np.median(modIQ.iloc[idx_base_start:idx_base_end])
    base_Uci   = np.median(Uci.iloc[idx_base_start:idx_base_end])
    base_Ucr   = np.median(Ucr.iloc[idx_base_start:idx_base_end])
    base_Ucav  = np.median(Ucav.iloc[idx_base_start:idx_base_end])
    #print(f'median modIQ : {base_modIQ}, median Uci : {base_Uci}, median Ucr : {base_Ucr}, median Ucav : {base_Ucav}')
    
    # Définition de la zone de recherche de pic (Autour de t0)
    # On cherche dans une petite fenêtre avant et après t0
    zone_start = max(0, min(t0_idx, zero_idx))
    zone_end   = zero_idx + 20
    
    # Extraction des comportements extrêmes dans cette zone
    max_modIQ_zone = modIQ.iloc[zone_start:zone_end].max()
    max_Ucr_zone = Ucr.iloc[zone_start:zone_end].max()
    max_Uci_zone = Uci.iloc[zone_start:zone_end].max()
    min_Uci_zone = Uci.iloc[zone_start:zone_end].min()
    max_Ucav_zone = Ucav.iloc[zone_start:zone_end].max()
    min_Ucav_zone = Ucav.iloc[zone_start:zone_end].min()
    
    # Calcul des variations réelles (L'événement par rapport au repos)
    delta_modIQ = max_modIQ_zone - base_modIQ
    delta_pos_Uci   = max_Uci_zone - base_Uci
    delta_neg_Uci   = min_Uci_zone - base_Uci
    delta_Ucr   = max_Ucr_zone - base_Ucr
    delta_pos_Ucav = max_Ucav_zone - base_Ucav
    delta_neg_Ucav = min_Ucav_zone - base_Ucav
    
    # Sécurité anti-division par zéro : 
    EPSILON = 0.01
    base_modIQ_safe = max(abs(base_modIQ), EPSILON)
    base_Uci_safe = max(abs(base_Uci), EPSILON)
    base_Ucr_safe = max(abs(base_Ucr), EPSILON)
    base_Ucav_safe = max(abs(base_Ucav), EPSILON)

    # Calcul des variations en POURCENTAGE
    pct_modIQ = (delta_modIQ / base_modIQ_safe) * 100.0
    pct_pos_Uci   = (delta_pos_Uci / base_Uci_safe) * 100.0
    pct_neg_Uci   = (delta_neg_Uci / base_Uci_safe) * 100.0
    pct_Ucr   = (delta_Ucr / base_Ucr_safe) * 100.0
    pct_pos_Ucav = (delta_pos_Ucav / base_Ucav_safe) * 100.0
    pct_neg_Ucav = (delta_neg_Ucav / base_Ucav_safe) * 100.0
    
    #print(f'index start : {zone_start} | index end : {zone_end}, start time value : {time[zone_start]} | end time value : {time[zone_end]}')
    #print(f'Deltas absolus -> modIQ: {delta_modIQ:.4f} | Uci_pos: {delta_pos_Uci:.4f} | Uci_neg: {delta_neg_Uci:.4f} | Ucr: {delta_Ucr:.4f} | Ucav_pos: {delta_pos_Ucav:.4f} | Ucav_neg: {delta_neg_Ucav:.4f}')
    #print(f'Variations (%) -> modIQ: {pct_modIQ:.1f}% | Uci_pos: {pct_pos_Uci:.1f}% | Uci_neg: {pct_neg_Uci:.4f}% | Ucr: {pct_Ucr:.1f}% | pct_pos_Ucav: {pct_pos_Ucav:.1f}% | pct_neg_Ucav: {pct_neg_Ucav:.4f}%')
    
    THRESHOLD_NOISE_PCT = 10.0
    THRESHOLD_PEAK_PCT = 90.00
    
    if pct_pos_Uci > THRESHOLD_NOISE_PCT:
        pct_Uci = pct_pos_Uci
    else:
        pct_Uci = pct_neg_Uci

    trend_Ucav_short = get_trend(time[t0_idx:t0_idx+5],Ucav.iloc[t0_idx:t0_idx+5])
   
    # Si modIQ monte plus que le bruit ambiant
    if pct_modIQ > THRESHOLD_NOISE_PCT:
        
        # pic de modIQ et Uci baisse significative
        if pct_modIQ > THRESHOLD_PEAK_PCT and pct_Uci < -THRESHOLD_NOISE_PCT:
            return "Power chain gain"
            
        # Montée significative de modIQ et Uci et Ucav baisse
        elif THRESHOLD_NOISE_PCT < pct_modIQ and pct_Uci > THRESHOLD_NOISE_PCT and trend_Ucav_short == -1:
            return "Beam loss"
        
        elif THRESHOLD_NOISE_PCT < pct_modIQ and pct_Uci > THRESHOLD_NOISE_PCT and trend_Ucav_short == 1:
            return "Beam loss;Pic Ucav"
            
    # Si aucune condition n'est remplie
    return "Other"

def classify_RF_protections(time, Ucav, Ucat):
    """
    Classifie le défaut "Dép seuil de sécurité RF" en comparant la valeur de la tension (Ucav) par rapport au seuil maximal (Ucat). 
    """
    # index du temps t=0 (le moment de la coupure)
    zero_idx = np.argmin(np.abs(time))
    
    # Définir la zone autour de 0
    start_idx = max(0, zero_idx - 20)
    end_idx   = min(len(time), zero_idx + 20)
    
    Ucat_fraction = float(Ucat) / 100.0

    # Calculer le ratio sur la fenêtre
    ratio = Ucav.iloc[start_idx:end_idx] / Ucat_fraction
     
    max_ratio = ratio.max()
    
    print(f"Index de t=0 : {zero_idx} | Ratio maximum mesuré : {max_ratio:.2f}")
    
    if max_ratio > 8.5:
        return "Ucav max"
    else:
        return "Pr max"

def classify_void(pickup, threshold_pickup=50):
    """
    Classifie le défaut "Seuil de vide" en mesurant la quantité de courant pickup par rapport au seuil. 
    """
    '''
    zero_idx = np.argmin(np.abs(pickup.index))
    # Définir la zone autour de 0
    start_idx = max(0, zero_idx - 30)
    end_idx   = min(len(pickup), zero_idx + 30)
    pickup_win = pickup.iloc[start_idx:end_idx]

    max_pickup = pickup_win.max()
    '''
    zero_idx = np.argmin(np.abs(pickup.index))
    pickup_win = pickup.iloc[0:zero_idx] #+ 38
    mean_pickup = pickup_win.mean() #+ 38
    #print(f'moyenne pickup : {mean_pickup}')

    if mean_pickup > threshold_pickup:
        return f"Ipu;{mean_pickup:.2f}uA"
    else:
        return f"Leak/outgassing;{mean_pickup:.2f}uA"

def classify_quench(Ucav):
    """
    Classifie le défaut "Claquage ou quench cavité" en mesurant le temps de chute 
    de la tension cavité (Ucav) de 100% à 50%.
    """
    # Trouver le début de la chute
    t0_idx, t0_time = detect_t0_zscore(Ucav)
    t0_idx = t0_idx - 1
    t0_time = Ucav.index[t0_idx]

    U_max = Ucav.iloc[t0_idx]
    
    # Définir la cible des 50%
    U_target = U_max * 0.5
    
    # Isoler la partie du signal qui se passe APRES le début de la chute
    Ucav_fall = Ucav.iloc[t0_idx:]
    
    # Chercher tous les points qui sont passés sous la barre des 50%
    below_50 = Ucav_fall[Ucav_fall < U_target]
    
    if below_50.empty:
        print("Alerte : Le signal ne descend jamais à 50% dans ce fichier.")
        return "Other;Incomplet (Pas de chute à 50%)"
        
    # Le temps "t_50" est le tout premier point de cette liste
    t_50 = below_50.index[0]
    
    # Calcul du Delta T
    delta_t = t_50 - t0_time
    
    print(f"U_max: {U_max:.2f} | t_start: {t0_time:.1f}µs | t_50: {t_50:.1f}µs | Chute en: {delta_t:.2f} µs")

    dt_microseconds = np.mean(np.diff(Ucav.index))
    THRESHOLD_BREAKDOWN = max(2.0, dt_microseconds * 1.05) # 2 µs
    THRESHOLD_FALSE_ALARM = 2000.0  # 2 ms (2000 µs)

    falltime = f";Temps de chute : {delta_t:.2f} us"

    if delta_t <= THRESHOLD_BREAKDOWN:
        return "Breakdown" + falltime
    elif delta_t > THRESHOLD_FALSE_ALARM:
        return "False alarm" + falltime
    else:
        return "Quench" + falltime

@chrono
def classify_file(file_name):
    """
    Classifie le fichier
    """
    with open(file_name, 'rb') as f:
        file_content = f.read()
    parameters, time, df_signals, df_defaults, _, header = Read_Signals(
        file_content, 
        compute_defauts=True, 
        compute_etats=True, 
        plot_signaux=False, 
        show_header=False
    )

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

    #Cas de tension presque nulle
    Ucav = df_signals.get("Ucav")
    zero_idx = np.argmin(np.abs(time))
    ucav_threshold = 0.1
    if np.mean(Ucav.iloc[0:zero_idx]) < ucav_threshold:
        return f'{default[0]};Other;Tension inférieur à {ucav_threshold}'

    match default[0]:
        case "Seuil pick-up":
            return default[0]
        case "Coupure externe rapide":
            return default[0]
        case "Absence autorisation RF":
            return default[0]
        case "Seuil de vide":
            pickup = df_signals.get("courant pickup")
            return default[0] + ";" + classify_void(pickup)
        case "Claquage ou quench cavité":
            return default[0] + ";" + classify_quench(Ucav)
        case "Dép seuil de sécurité RF":
            Ucat = parameters.get("UCAT")
            return default[0] + ";" + classify_RF_protections(time, Ucav, Ucat)
        case "Rég signal RF hors tolérance":
            Uci = df_signals.get("Uci")
            Ucr = df_signals.get("A Ucr") 
            modIQ = df_signals.get("IQ")
            return default[0] + ";"+ classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ)
        case _:
            return "Other"

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
