import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import linregress

def _verify_zero_crossings(signal, dominant_freq, dt, peak_amp):
    """
    Vérifie si le signal traverse son point d'équilibre un nombre de fois
    cohérent avec sa fréquence dominante estimée.
    """
    n = len(signal)
    period_points = int(1.0 / (dominant_freq * dt))
    
    # Fenêtre pour centrer le signal sur zéro
    zcross_window_size = max(5, 3 * period_points)
    zcross_baseline = pd.Series(signal).rolling(window=zcross_window_size, center=True, min_periods=1).mean().to_numpy()
    sig_zcross = signal - zcross_baseline

    # Seuil d'hystérésis pour ignorer le micro-bruit lors du passage par zéro
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
    
    # Validation : au moins 2 croisements, et au moins 45% de ce que la FFT prédisait
    is_crossing_valid = (actual_crossings >= 2) and (actual_crossings >= 0.45 * expected_crossings)

    return is_crossing_valid, actual_crossings, expected_crossings, sig_zcross, period_points

def _verify_oscillation_survival(sig_zcross, period_points, min_physical_amp):
    """
    Vérifie si l'oscillation persiste jusqu'à la fin de la fenêtre d'analyse 
    en étudiant l'enveloppe du signal.
    """
    n = len(sig_zcross)
    env_window = max(5, int(1.5 * period_points))
    envelope = pd.Series(np.abs(sig_zcross)).rolling(window=env_window, center=True, min_periods=1).max().to_numpy()
    active_envelope = envelope[envelope > min_physical_amp]

    is_touching_end = False
    
    if len(active_envelope) > 0:
        # On utilise le 75ème centile pour trouver l'amplitude typique des vagues.
        typical_env = np.percentile(active_envelope, 75)
        
        # Le seuil de survie est fixé à 25% de cette amplitude typique
        alive_threshold = max(min_physical_amp, 0.25 * typical_env)
        alive_indices = np.where(envelope >= alive_threshold)[0]
        
        if len(alive_indices) > 0:
            last_alive_idx = alive_indices[-1]
            distance_to_end = n - last_alive_idx
            
            # Écart toléré entre la fin de l'oscillation et la chute de tension
            max_distance_allowed = max(200, int(1.0 * period_points))

            is_touching_end = (distance_to_end <= max_distance_allowed)
            
            # Affichage
            # print(f"Typical Env (75%): {typical_env:.4f} | Seuil de vie: {alive_threshold:.4f}")
            # print(f"Dernier vivant: {last_alive_idx}/{n} | Distance: {distance_to_end} (Max permis: {max_distance_allowed})")
            
    return is_touching_end

def analyze_oscillation(signal, dt, snr_threshold=8.5, min_physical_amp=0.005):
    """
    Détermine si le signal oscille et renvoie la fréquence
    - snr_threshold : Rapport signal/bruit dans le spectre (relatif).
    - min_physical_amp : Amplitude physique minimale en dessous de laquelle on considère que c'est juste du bruit.
    """
    if len(signal) < 10:    
        return False, 0.0 

    n = len(signal)
    sig_array = np.asarray(signal)
    
    dyn_window = max(5, int(n * 0.20))
    baseline = pd.Series(sig_array).rolling(window=dyn_window, center=True, min_periods=1).median().to_numpy()
    sig_detrended = sig_array - baseline

    temporal_amp = np.percentile(np.abs(sig_detrended), 99)

    # Fenêtrage
    window = np.hanning(n)
    sig_windowed = sig_detrended * window
    
    yf = np.abs(fft(sig_windowed))
    xf = fftfreq(n, d=dt)

    # Normalisation
    yf = 2.0 / n * yf
    freq_resolution = 1.0 / (n * dt)
    min_valid_freq = 0.5 * freq_resolution

    mask = xf > min_valid_freq
    if not np.any(mask):
        return False, 0.0

    xf_pos, yf_pos = xf[mask], yf[mask]

    peaks_indices, _ = find_peaks(yf_pos)
    if len(peaks_indices) > 0:
        best_peak_idx = np.argmax(yf_pos[peaks_indices])
        max_idx = peaks_indices[best_peak_idx]
    else:
        max_idx = np.argmax(yf_pos)
    
    dominant_freq = xf_pos[max_idx]
    peak_amp = yf_pos[max_idx]

    # Vérification d'amplitude
    noise_level = snr_threshold * np.median(yf_pos)
    is_above_noise = peak_amp > noise_level
    is_physically_significant = temporal_amp > min_physical_amp

    # Verification zero crossing
    is_crossing_valid, act_cross, exp_cross, sig_zcross, period_points = _verify_zero_crossings(signal, dominant_freq, dt, peak_amp)

    # Verification si l'oscillation est toujours active à la fin
    is_touching_end = _verify_oscillation_survival(sig_zcross, period_points, min_physical_amp)

    # Verdict final 
    is_oscillating = is_above_noise and is_physically_significant and is_crossing_valid and is_touching_end

    # Affichage
    #print(f'snr threshold : {snr_threshold}, min physical amp : {min_physical_amp}')
    #print(f'is_above_noise : {is_above_noise} | is_physically_significant : {is_physically_significant} | is_crossing_valid : {is_crossing_valid} | is_touching_end: {is_touching_end}')
    #print(f"{signal.name} Freq: {dominant_freq:.0f}Hz | Amp: {temporal_amp:.5f} | Amp fft: {peak_amp:.5f} | Spectral noise: {noise_level:.5f} | Z-Cross: {act_cross}/{int(exp_cross)} | is_oscillating: {is_oscillating}")
    
    return is_oscillating, dominant_freq

def get_trend(time, signal, tolerance=1e-5):
    """
    Détermine si le signal est croissant, décroissant ou stable
    en utilisant la pente de la régression linéaire.
    Retourne 1 si le signal est croissant, -1 si il est décroissant et 0 sinon
    La tolérance permet d'ignorer le bruit plat.
    """
    if signal.size < 2 or time.size < 2:
        signal_name = getattr(signal, 'name', 'Unknown signal')
        print(f"{signal_name} | array too small (len={len(signal)})")
        return 0

    slope = np.polyfit(time, signal, 1)[0]

    if slope > tolerance:
        trend = 1
    elif slope < -tolerance:
        trend =  -1
    else:
        trend = 0

    #signal_name = getattr(signal, 'name', 'Unknown signal')
    #print(f'{signal_name} | slope : {slope:.7f} | trend : {trend}')
    return trend

def detect_t0_zscore(signal, z_threshold=5.0, calm = True):
    """
    Détecte t0, le moment ou le signal signal sort de son amplitude moyenne avec un Z-score.
    - z_threshold : Nombre d'écarts-types pour déclencher l'alarme (5 est standard).
    """
    zero_idx = np.argmin(np.abs(signal.index))

    # Baseline et zone sûre
    signal_mean = signal.rolling(window=1000, min_periods=1).mean() 
    end_safe_zone = max(100, zero_idx - 20000) 
    safe_zone = signal.iloc[0:end_safe_zone]
    search_window = min(10000, len(safe_zone))

    # Z-score d'amplitude
    all_stds_amp = safe_zone.rolling(window=search_window, min_periods=int(search_window*0.5)).std()
    signal_std = max(all_stds_amp.min(), 1e-6)
    z_scores_amp = abs((signal - signal_mean) / signal_std)

    # Z-score différentiel (dérivée)
    signal_diff = signal.diff(periods=20).fillna(0)
    safe_zone_diff = signal_diff.iloc[0:end_safe_zone]

    all_stds_diff = safe_zone_diff.rolling(window=search_window, min_periods=int(search_window*0.5)).std()
    all_means_diff = safe_zone_diff.rolling(window=search_window, min_periods=int(search_window*0.5)).mean()
    
    best_idx_diff = all_stds_diff.idxmin()
    diff_std = max(all_stds_diff[best_idx_diff], 1e-6)
    diff_mean = all_means_diff[best_idx_diff]
    
    z_scores_diff = abs((signal_diff - diff_mean) / diff_std)

    # Combiner les deux Z-scores (en gardant la pire anomalie pour chaque point)
    z_scores = np.maximum(z_scores_amp, z_scores_diff)

    # Détection d'anomalies (Points consécutifs)
    target_z_scores = z_scores.iloc[0:zero_idx + 40]
    min_consecutive = 3 
    
    is_anomaly_bin = (target_z_scores > z_threshold).astype(int)

    future_window = is_anomaly_bin.iloc[::-1].rolling(window=min_consecutive).min().iloc[::-1]
    valid_anomaly_starts = (future_window == 1)

    #On vérifie si les N prochains points sont anormaux en regardant vers l'avant avec un décalage négatif
    valid_anomaly_starts = is_anomaly_bin.rolling(window=min_consecutive).min().shift(-(min_consecutive - 1)) == 1
    
    anomalies = target_z_scores[valid_anomaly_starts]

    # On extrait uniquement les Z-scores qui démarrent une vraie chute
    if anomalies.empty:
        return zero_idx-10000, None

    # Le filtre calme
    if calm:
        signal_median = safe_zone.median()
        survival_threshold = signal_median * 0.5 
        target_ucav = signal.iloc[0:zero_idx + 40]
        
        # Un point est sain seulement si le Z-score est bon et que la cavité est allumée
        is_zscore_ok = (target_z_scores <= z_threshold)
        is_amplitude_ok = (np.abs(target_ucav) > survival_threshold)
        is_healthy = (is_zscore_ok & is_amplitude_ok).astype(int)
        
        # Un vrai calme c'est quand le signal est sain sur au moins 200 points consécutifs.
        true_calm = is_healthy.rolling(window=200, min_periods=1).min() == 1.0
        last_calm_times = true_calm[true_calm]
        
        if not last_calm_times.empty:
            end_of_last_calm = last_calm_times.index[-1]
            
            # Le vrai t0 est la première anomalie qui arrive APRÈS cette ultime paix
            final_anomalies = anomalies[anomalies.index > end_of_last_calm]
            
            if not final_anomalies.empty:
                t0_time = final_anomalies.index[0]
                t0_idx = signal.index.get_loc(t0_time)
                return t0_idx, t0_time

    # Repli si aucun filtre de calme ou si le filtre n'a rien trouvé
    t0_time = anomalies.index[0]
    t0_idx = signal.index.get_loc(t0_time)
    
    return t0_idx, t0_time

def find_true_t0_with_ampt(Ucav, zero_idx, ampt):
    """
    Calcule le véritable début de la chute de tension (t0_ampt_idx) en se basant 
    sur le paramètre de déclenchement AMPT du fichier.
    """
    # Si ampt est trop bas l'ecart entre t0_ampt_idx et zero_idx est négligeable
    if ampt <= 3.00:
        return zero_idx, None

    ampt_val = ampt / 100.0
    if ampt_val <= 0 or ampt_val >= 1:
        ampt_val = 0.03

    trigger_val = Ucav.iloc[zero_idx]
    signal_before_zero = Ucav.iloc[:zero_idx]
    
    # Définition de la ligne de base (plateau réel avant l'événement)
    idx_base_start = max(0, zero_idx - 1000)
    idx_base_end = max(1, zero_idx - 300)
    real_plateau = Ucav.iloc[idx_base_start:idx_base_end].median()

    # Détection du type de défaut (Chute vs Montée)
    is_drop = trigger_val < real_plateau

    if is_drop:
        #Cas chute
        plateau_ref = trigger_val / (1.0 - ampt_val)
        threshold = plateau_ref * 0.99
        # On cherche les points où le signal était encore "en haut"
        valid_indices = np.where(signal_before_zero >= threshold)[0]
        
    else:
        # Cas montée
        plateau_ref = real_plateau
        threshold = plateau_ref * 1.01
        # On cherche les points où le signal était encore "en bas"
        valid_indices = np.where(signal_before_zero <= threshold)[0]

    # Validation de l'index trouvé
    max_distance_allowed = 6000

    if len(valid_indices) > 0:
        t0_ampt_idx = valid_indices[-1]
        
        #Si l'index trouver est trop éloigné, c'est qu'il n'y a pas de décalage signifivatif.
        if (zero_idx - t0_ampt_idx) > max_distance_allowed:
            # print(f"Alerte : t0_ampt_idx trop éloigné ({zero_idx - t0_ampt_idx} pts). Annulation.")
            t0_ampt_idx = zero_idx
    else:
        t0_ampt_idx = zero_idx

    #print(f'ampt_val : {ampt_val}')
    #print(f'trigger_val : {trigger_val}') 
    #print(f'plateau_ref : {plateau_ref}')
    
    return t0_ampt_idx, plateau_ref

def detect_end_of_staircase(signal, zero_idx, dt_us):
    """
    Détecte la fin d'une forme en escalier dans un signal avant une chute.
    Retourne l'index numérique de la fin de l'escalier, ou None si aucun n'est détecté.
    """
    safety_margin_pts = int(10000 / dt_us)

    if zero_idx - safety_margin_pts <= 0:
        return None

    signal_before_drop = signal.iloc[:zero_idx - safety_margin_pts]
    n_points = len(signal_before_drop)

    if n_points < 100:
        return None

    # Vérification de l'épaisseur du bruit (Filtre anti-oscillation)
    check_pts = max(5, int(2000 / dt_us))
    local_max = signal_before_drop.rolling(window=check_pts).max()
    local_min = signal_before_drop.rolling(window=check_pts).min()
    local_thickness = local_max - local_min

    # Si l'épaisseur médiane dépasse le seuil, ce n'est pas plat : c'est une oscillation
    if local_thickness.median() > 0.10:
        return None

    # Lissage du signal
    window_us = 5000 
    window_pts = max(3, int(window_us / dt_us))
    smoothed_series = signal_before_drop.rolling(window=window_pts, center=True, min_periods=1).mean()

    # Conversion en tableau NumPy pur pour éviter les plantages avec SciPy
    smoothed_array = smoothed_series.to_numpy()

    # Détection des creux (Valleys)
    wave_amplitude_threshold = 0.3

    # On cherche les creux sur le signal inversé
    valleys, _ = find_peaks(-smoothed_array, prominence=wave_amplitude_threshold)
    if len(valleys) >= 2:
        return None 

    # Analyse de la dérivé (Marche de l'escalier)
    step = max(1, int(5000 / dt_us))
    derivative = smoothed_series.diff(periods=step).fillna(0).to_numpy()

    min_step_height = 0.04
    min_dist_pts = int(20000 / dt_us)

    # Recherche des pics de la dérivée (les montées de marches)
    rising_peaks, _ = find_peaks(derivative, height=min_step_height, distance=min_dist_pts)

    min_fall_height = 0.20 

    # Recherche des descentes (pics sur la dérivée inversée)
    falling_peaks, _ = find_peaks(-derivative, height=min_fall_height, distance=min_dist_pts)

    # Si on trouve plusieurs vraies grandes descentes, c'est de l'oscillation
    if len(falling_peaks) >= 2:
        return None

    # Conclusion et positionnement
    min_steps_required = 3

    if len(rising_peaks) > min_steps_required:
        last_peak_idx = rising_peaks[-1]

        # On se place juste un peu après la fin de la dernière montée (sur le plat)
        end_stair_idx = last_peak_idx + int(window_pts / 2)
        end_stair_idx = min(end_stair_idx, len(signal_before_drop) - 1)

        return end_stair_idx
        
    return None

def calculate_fall_time(Ucav, real_t0_idx, target_pct=0.5,z_threshold = 5):
    """
    Calcule le temps mis par le signal pour chuter à un certain pourcentage de son amplitude maximale.
    Retourne un tuple : (delta_t, u_max, t_start, t_target)
    Retourne None si le signal ne chute jamais à la valeur cible.
    """
    n_points = len(Ucav)
    start_idx = max(0, real_t0_idx - 100)
    end_idx = min(n_points, real_t0_idx + 30)

    search_window = Ucav.iloc[start_idx:end_idx]

    # Appel de la fonction z-score sur la petite fenêtre
    local_t0_idx, _ = detect_t0_zscore(search_window / 10.0, z_threshold=z_threshold)

    # Repositionnement de l'index local vers l'index global du fichier
    global_t0_idx = start_idx + local_t0_idx - 1
    global_t0_idx = max(0, min(n_points - 1, global_t0_idx))

    t0_time = Ucav.index[global_t0_idx]

    u_max = Ucav.iloc[global_t0_idx]
    u_target = u_max * target_pct
    
    ucav_fall = Ucav.iloc[global_t0_idx:]
    below_target_points = ucav_fall[ucav_fall < u_target]
    
    if below_target_points.empty:
        return None
        
    t_target = below_target_points.index[0]
    delta_t = t_target - t0_time
    
    return delta_t, u_max, t0_time, t_target

def detect_uci_rise_start(Uci, t0_ampt_idx, dt_us, start_limit_idx=0):
    """
    Détecte l'index du véritable début de la grande montée du signal Uci,
    en cherchant le moment où le signal quitte son niveau de base.
    """
    search_window = Uci.iloc[start_limit_idx:t0_ampt_idx]
    
    if len(search_window) < 1000:
        return max(0, t0_ampt_idx - int(5000 / dt_us))
        
    # Lissage robuste
    window_pts = max(5, int(2000 / dt_us))
    smoothed_uci = search_window.rolling(window=window_pts, center=True, min_periods=1).mean()
    
    valid_smoothed = smoothed_uci.dropna()
    if valid_smoothed.empty:
        return max(0, t0_ampt_idx - int(5000 / dt_us))
        
    # Calcul des niveaux
    base_val = np.percentile(valid_smoothed, 2)   
    peak_val = np.percentile(valid_smoothed, 98)  
    amplitude = peak_val - base_val
    
    if amplitude < 1.0:
        return max(0, t0_ampt_idx - int(5000 / dt_us))
        
    # Définition du fil de détente
    threshold_val = base_val + 0.05 * amplitude
    
    # Recherche à rebours
    peak_idx_label = valid_smoothed[valid_smoothed >= peak_val].index[0]
    peak_pos = search_window.index.get_loc(peak_idx_label)
    
    before_peak = smoothed_uci.iloc[:peak_pos]
    
    below_threshold_mask = before_peak < threshold_val
    below_indices = np.where(below_threshold_mask)[0]
    
    if len(below_indices) > 0:
        start_idx = start_limit_idx + below_indices[-1]
        return start_idx
        
    return start_limit_idx