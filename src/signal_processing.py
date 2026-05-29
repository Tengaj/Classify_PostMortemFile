import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import linregress

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

def detect_end_of_staircase(Ucav, zero_idx, dt_us):
    """
    Détecte la fin d'une forme en escalier dans un signal avant une chute.
    
    Retourne:
        end_stair_idx : L'index où l'escalier se termine et le plateau commence.
                        (Retourne None si aucun escalier n'est détecté).
    """
    safety_margin_pts = int(500 / dt_us)
    if zero_idx - safety_margin_pts <= 0:
        return None # Signal trop court
    signal_before_drop = Ucav.iloc[:zero_idx - safety_margin_pts]
    n = len(signal_before_drop)
    if n < 100:
        return None

    # Lissage (Moyenne glissante) pour éliminer le bruit
    window_us = 5000 
    window_pts = max(3, int(window_us / dt_us))
    smoothed_signal = signal_before_drop.rolling(window=window_pts, center=True, min_periods=1).mean()

    # Un escalier est "monotone" (il monte). Une oscillation fait des grosses vagues.
    # On cherche si le signal lui-même possède des "creux" (valleys) d'au moins 0.3 Volt de profondeur.
    wave_amplitude_threshold = 0.3
    
    # On cherche les creux sur le signal inversé
    valleys, _ = find_peaks(-smoothed_signal, prominence=wave_amplitude_threshold)
    
    if len(valleys) >= 2:
        return None 

    # Calcul de la dérivée (différence entre un point et le précédent)
    step = max(1, int(2000 / dt_us)) # On regarde la différence sur 2000µs
    derivative = smoothed_signal.diff(periods=step).fillna(0).to_numpy()

    # Recherche des pics de la dérivée (les montées de marches)
    # Hauteur minimum en deux marches
    min_step_height = 0.05 
    
    # L'écart minimum entre deux marches
    min_dist_pts = int(20000 / dt_us) 

    rising_peaks, _ = find_peaks(derivative, height=min_step_height, distance=min_dist_pts)

    # On exige une chute beaucoup plus grande pour considérer que c'est une oscillation
    # Cela permet d'ignorer les petits "creux" ou le bruit sur les plats de l'escalier.
    min_fall_height = 0.20 
    
    # Recherche des descentes (pics sur la dérivée inversée)
    falling_peaks, _ = find_peaks(-derivative, height=min_fall_height, distance=min_dist_pts)
    
    # Si on trouve plusieurs VRAIES grandes descentes, c'est de l'oscillation
    if len(falling_peaks) >= 2:
        return None

    # Exigence d'un nombre minimum de marches
    min_steps_required = 3

    # Déterminer la fin de l'escalier
    if len(rising_peaks) > min_steps_required:
        # La fin de l'escalier est juste après la dernière montée détectée.
        # On ajoute la moitié de la fenêtre de lissage pour être "sur le plat".
        last_peak_idx = rising_peaks[-1]
        
        # On se place juste un peu après la fin de la dernière montée
        end_stair_idx = last_peak_idx + int(window_pts / 2)
        
        # Sécurité : s'assurer qu'on ne dépasse pas la zone qu'on analysait
        end_stair_idx = min(end_stair_idx, len(signal_before_drop) - 1)
        
        return end_stair_idx
    else:
        # Aucun escalier détecté (ou pas assez de marches)
        return None