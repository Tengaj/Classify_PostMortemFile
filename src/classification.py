import numpy as np
import pandas as pd
from PyPostMortem import Read_Signals
from utils import chrono, CONFIG
from signal_processing import (
    analyze_oscillation, 
    get_trend, 
    detect_t0_zscore,
    find_true_t0_with_ampt, 
    detect_end_of_staircase,
    calculate_fall_time,
    detect_uci_rise_start
)

def _check_instability_oscillations(Uci, modIQ, Ucav, dt_seconds, t0_zscore_idx, t0_ampt_idx, end_stair_idx):
    """
    [Sous-fonction] Vérifie la présence d'oscillations cohérentes pour classifier les défauts de type "Taconis" ou "Loop instability".
    """
    cfg = CONFIG.get("thresholds", {}).get("ecav_instability", {})
    
    amp_uci = cfg.get("min_physical_amp_uci", 0.002)
    amp_modiq = cfg.get("min_physical_amp_modiq", 0.1)
    amp_ucav = cfg.get("min_physical_amp_ucav", 0.0001)
    taconis_max_hz = cfg.get("taconis_max_hz", 100)
    
    # Définition de la fenêtre d'analyse
    # Si le décrochage (z-score) est détecté APRÈS le défaut (ampt), cela signifie que le signal oscillait potentiellement depuis le début.
    if t0_zscore_idx > t0_ampt_idx:
        if end_stair_idx is not None:
            # S'il y a un escalier, on analyse juste après
            start_osci = end_stair_idx
        else:
            # Sinon, on cherche t0 sans le filtre de "calme"
            start_osci, _ = detect_t0_zscore(Ucav, calm=False)
            if start_osci > t0_ampt_idx:
                start_osci = max(0, t0_ampt_idx - 10000)
        end_osci = t0_ampt_idx
    else:
        start_osci = t0_zscore_idx
        end_osci = t0_ampt_idx + 2

    # Sécurité : on vérifie que la fenêtre est valide (start avant end)
    if start_osci >= end_osci:
        return None

    # print(f'Osci window : {Ucav.index[start_osci]}, {Ucav.index[end_osci]}')

    # Analyse des oscillations pour chaque signal
    osc_Uci, f_Uci     = analyze_oscillation(
    	Uci.iloc[start_osci:end_osci],   dt_seconds, min_physical_amp=amp_uci
    )
    osc_modIQ, f_modIQ = analyze_oscillation(
    	modIQ.iloc[start_osci:end_osci], dt_seconds, min_physical_amp=amp_modiq
    )
    osc_Ucav, f_Ucav   = analyze_oscillation(
    	Ucav.iloc[start_osci:end_osci],  dt_seconds, min_physical_amp=amp_ucav
    )
    
    # Verdict de cohérence et classification
    # Il faut que les trois signaux oscillent de manière significative
    if osc_Uci and osc_modIQ and osc_Ucav:
        frequencies = [f_Uci, f_modIQ, f_Ucav]
        f_max = max(frequencies)
        f_min = max(min(frequencies), 1e-6)  # Sécurité division par zéro
        
        ratio = f_max / f_min
        # print(f'Ratio : {ratio}')
        
        is_coherent = False
        
        # Cas A : Fondamentale (les fréquences sont quasi-identiques)
        if ratio <= 1.5:
            is_coherent = True
            
        # Cas B : Harmonique (Le ratio correspond à la 2ème, 3ème, 4ème ou 5ème harmonique)
        else:
            closest_integer = round(ratio)
            if 2 <= closest_integer <= 5 and abs(ratio - closest_integer) <= 0.2:
                is_coherent = True
        
        # Classification finale si les oscillations sont liées physiquement
        if is_coherent:
            if f_Ucav > taconis_max_hz:
                return "Loop instability" # f"Loop instability;{f_Ucav:.0f}Hz"
            else:
                return "Taconis"          # f"Taconis;{f_Ucav:.0f}Hz"
        else:
            return "Oscillation"
            
    return None

def _check_quench_signatures(time, Ucav, Uci, Ucr, t0_zscore_idx, start_idx_Uci_rise, end_stair, t0_ampt_idx):
    """
    [Sous-fonction] Analyse les tendances de Uci, Ucr et le temps de chute de Ucav
    pour détecter un Quench ou un Quench aside.
    """
    cfg = CONFIG.get("thresholds", {}).get("ecav_instability", {})
    
    target_pct = cfg.get("fall_target_pct", 1.0) / 100.0
    min_false_alarm_us = cfg.get("min_false_alarm_us", 10000.0)

    # Filtre montée trop précoce
    zero_idx = np.argmin(np.abs(time))
    limit_early_rise = int(zero_idx * 0.5)
    
    if start_idx_Uci_rise < limit_early_rise:
        # print(f"Montée Uci trop précoce (Index {start_idx_Uci_rise} < Limite {limit_early_rise}). Rejet du Quench.")
        return None
        
    # Définitions des fenêtres
    start_idx_large = start_idx_Uci_rise

    t_win_large   = time[start_idx_large:t0_ampt_idx]
    Uci_win_large = Uci.iloc[start_idx_large:t0_ampt_idx]
    Ucr_win_large = Ucr.iloc[start_idx_large:t0_ampt_idx]

    # Filtre : amplitude de la montée Uci
    base_start = max(0, start_idx_large - 500)
    base_end   = max(1, start_idx_large - 100)
    
    base_Uci  = np.median(Uci.iloc[base_start:base_end])
    delta_Uci = np.median(Uci_win_large) - base_Uci 
    
    # Sécurité contre la division par zéro
    base_Uci_safe = max(abs(base_Uci), 0.01)
    ratio_Uci = (delta_Uci / base_Uci_safe) * 100

    if ratio_Uci < 10.0:
        # print(f"Ratio d'augmentation Uci trop bas ({ratio_Uci:.1f}%). Pas de quench.")
        return None

    # Analyse des tendances globales
    trend_Uci = get_trend(t_win_large, Uci_win_large, tolerance=1e-5)
    trend_Ucr = get_trend(t_win_large, Ucr_win_large, tolerance=1e-6)

    # Calcul de la chute de la tension
    quench = False
    
    result = calculate_fall_time(Ucav, zero_idx, target_pct=target_pct)
    
    if result is None:
        # print(f"Alerte : Le signal ne descend jamais à {100 - target_pct * 100}% dans ce fichier.")
        return None

    delta_t, u_max, t0_time, t = result
    # print(f"U_max: {u_max:.2f} | t_start: {t0_time:.1f}µs | t: {t:.1f}µs | Chute en: {delta_t:.2f} µs")

    dt_us = np.mean(np.diff(time))
    threshold_breakdown = max(2.0, dt_us * 2.0) # ~11 µs
    
    if threshold_breakdown < delta_t < min_false_alarm_us:
        quench = True

    # Détection des pics brutaux (spikes) sur Uci
    step_points = max(1, int(50.0 / dt_us))
    pts_150     = max(1, int(150.0 / dt_us)) 
    
    total_rise_Uci = Uci_win_large.max() - Uci_win_large.min()
    diff_50        = Uci_win_large.diff(periods=step_points)
    max_jump_Uci   = diff_50.max()
    
    is_sudden_spike = False
    
    if total_rise_Uci > 0.5:
        if max_jump_Uci > 0.65 * total_rise_Uci:
            
            jump_idx = np.nanargmax(diff_50)
            
            idx_before = max(0, jump_idx - pts_150)
            idx_after  = min(len(Uci_win_large) - 1, jump_idx + pts_150)
            
            val_before = Uci_win_large.iloc[idx_before]
            val_after  = Uci_win_large.iloc[idx_after]
            
            true_shift = val_after - val_before
            
            # Si le décalage durable représente au moins 40% du saut
            if true_shift > 0.40 * max_jump_Uci:
                is_sudden_spike = True
                # print(f"Pic brutal, Hauteur: {total_rise_Uci:.2f} | Saut: {max_jump_Uci:.2f} | Décalage durable: {true_shift:.2f}")

    # Verdict final
    if trend_Uci == 1:
        if is_sudden_spike:
            return None 
        if quench:
            return "Quench"
        elif trend_Ucr == 1:
            return "Quench aside"
            
    return None
    
def _check_amplitude_anomalies(time, Uci, Ucav, modIQ, beam_presence, t0_ampt_idx, t0_zscore_idx):
    """
    [Sous-fonction] Analyse les pics anormaux par rapport au bruit de fond (Beam loss, Power chain gain).
    """
    # Zone du calcul du bruit de fond
    idx_base_start = max(0, t0_ampt_idx - 1000)
    idx_base_end = max(1, t0_ampt_idx - 100)

    # Zone de recherche du pic
    zone_start = t0_ampt_idx - 20
    zone_end   = t0_ampt_idx + 30
    
    # Calcul du bruit de fond
    base_window_modIQ = modIQ.iloc[idx_base_start:idx_base_end]
    base_window_Uci   = Uci.iloc[idx_base_start:idx_base_end]
    
    base_modIQ = np.median(base_window_modIQ)
    base_Uci   = np.median(base_window_Uci)

    # Épaisseur robuste du bruit de fond (entre le 5e et 95e percentile)
    noise_thickness = np.percentile(base_window_Uci, 95) - np.percentile(base_window_Uci, 5)
    
    # Mesure des êxtremes dans la zone de défaut
    max_modIQ_zone = modIQ.iloc[zone_start:zone_end].max()
    max_Uci_zone = Uci.iloc[zone_start:zone_end].max()
    min_Uci_zone = Uci.iloc[zone_start:zone_end].min()
    
    # Deltas absolus
    delta_modIQ = max_modIQ_zone - base_modIQ
    delta_pos_Uci = max_Uci_zone - base_Uci
    delta_neg_Uci = min_Uci_zone - base_Uci
    
    # Protection division par zéro  
    EPSILON = 0.01
    base_modIQ_safe = max(abs(base_modIQ), EPSILON)
    base_Uci_safe = max(abs(base_Uci), EPSILON)

    pct_modIQ = (delta_modIQ / base_modIQ_safe) * 100.0
    pct_pos_Uci = (delta_pos_Uci / base_Uci_safe) * 100.0
    pct_neg_Uci = (delta_neg_Uci / base_Uci_safe) * 100.0
    
    #print(f'index start : {zone_start} | index end : {zone_end}, start time value : {time[zone_start]} | end time value : {time[zone_end]}')
    #print(f'Deltas absolus -> modIQ: {delta_modIQ:.4f} | Uci_pos: {delta_pos_Uci:.4f} | Uci_neg: {delta_neg_Uci:.4f}')
    #print(f'Variations (%) -> modIQ: {pct_modIQ:.1f}% | Uci_pos: {pct_pos_Uci:.1f}% | Uci_neg: {pct_neg_Uci:.4f}%')
    #print(f'noise_thickness : {noise_thickness}')
    
    # Analyse et désicion
    cfg = CONFIG.get("thresholds", {}).get("ecav_instability", {})
    threshold_noise_pct = cfg.get("threshold_noise_pct", 10.0)
    threshold_peak_pct  = cfg.get("threshold_peak_pct", 90.0)

    # Un vrai pic doit dépasser significativement le bruit ambiant
    is_real_peak = delta_pos_Uci > (noise_thickness)

    # Sélection du pourcentage Uci pertinent (positif si vrai pic, sinon on garde le creux négatif)
    if pct_pos_Uci > threshold_noise_pct and is_real_peak:
        pct_Uci = pct_pos_Uci
    else:
        pct_Uci = pct_neg_Uci
    
    # Tendance très courte sur Ucav au moment exact du décrochage (sécurisé contre les index négatifs)
    trend_start = max(0, t0_zscore_idx - 1)
    trend_end   = t0_zscore_idx + 5
    trend_Ucav_short = get_trend(time[trend_start:trend_end], Ucav.iloc[trend_start:trend_end], tolerance=1e-6)
    #print(f'pct_Uci : {pct_Uci}, is_real_peak : {is_real_peak}, trend_Ucav_short : {trend_Ucav_short}')
   
    if pct_modIQ > threshold_noise_pct:

        # Power chain gain, gros pic modIQ, suivit d'une chute de Uci
        if pct_modIQ > threshold_peak_pct and pct_Uci < -threshold_noise_pct:
            return "Power chain gain"

        # Beam loss (Pic modIQ et Uci, chute de Ucav et presence du faisceau)
        elif pct_Uci > threshold_noise_pct and is_real_peak and trend_Ucav_short == -1:
            start_idx = max(0, t0_zscore_idx - 20)
            end_idx = max(1, t0_zscore_idx + 10)
            
            beam_window = beam_presence.iloc[start_idx:end_idx]
            
            if len(beam_window) > 0:
                # On compte le nombre exact de points où le faisceau est présent (état = 1)
                points_on = beam_window.sum()
                
                # Le faisceau est présent au moins 2 points
                if points_on >= 2:
                    return "Beam loss"

        # Beam loss mais montée de Ucav
        #elif pct_Uci > threshold_noise_pct and trend_Ucav_short == 1:
            #return "Other,Faulty beam presence"
    
    return None

def _check_beam_anomalies(beam_presence, t0_zscore_idx):
    """
    [Sous-fonction] Vérifie si le faisceau présente des instabilités autour du moment du défaut.
    """
    start_idx = max(0, t0_zscore_idx - 20)
    end_idx = max(1, t0_zscore_idx + 10)
    
    beam_window = beam_presence.iloc[start_idx:end_idx]
    
    if len(beam_window) > 0:
        # On compte le nombre exact de points où le faisceau est présent (état = 1)
        points_on = beam_window.sum()
        total_points = len(beam_window)
        
        # Le faisceau est présent au moins 2 points mais il n'est pas parfaitement stable
        if points_on >= 2 and points_on < total_points:
            return "Beam related anomaly"
            
    return None

def _check_detuning_anomalies(time, Uci, Ucr, start_idx_Uci_rise, t0_ampt_idx):
    """
    [Sous-fonction] Vérifie si Uci et Ucr montent continuellement depuis le début du fichier.
    Caractéristique des dérives lentes.
    """
    #print(f"start_idx_Uci_rise : {start_idx_Uci_rise}")
    zero_idx = np.argmin(np.abs(time))
    
    limit_early_rise = int(zero_idx * 0.5)

    #start_idx_Uci_rise = 0
    
    # Si la montée commence tardivement, on annule.
    if start_idx_Uci_rise >= limit_early_rise:
        return None

    # Définition de la fenêtre d'analyse
    t_win = time[start_idx_Uci_rise:t0_ampt_idx]
    Uci_win = Uci.iloc[start_idx_Uci_rise:t0_ampt_idx]
    Ucr_win = Ucr.iloc[start_idx_Uci_rise:t0_ampt_idx]

    # Analyse des tendance
    trend_Uci = get_trend(t_win, Uci_win, tolerance=1e-6)
    trend_Ucr = get_trend(t_win, Ucr_win, tolerance=1e-6)

    # Si les deux signaux montent
    if trend_Uci == 1 and trend_Ucr == 1:
        return "Detuning control error"
        
    return None

def _check_other_peak(Uci, modIQ, t0_ampt_idx):
    """
    [Sous fonction] Vérifie la présence de pic autour du moment du défaut sur les signaux Uci et modIQ.
    """
    # Zone du calcul du bruit de fond
    idx_base_start = max(0, t0_ampt_idx - 1000)
    idx_base_end = max(1, t0_ampt_idx - 100)

    # Zone de recherche du pic
    zone_start = t0_ampt_idx - 30
    zone_end   = t0_ampt_idx + 30
    
    # Calcul du bruit de fond
    base_window_modIQ = modIQ.iloc[idx_base_start:idx_base_end]
    base_window_Uci   = Uci.iloc[idx_base_start:idx_base_end]
    
    base_modIQ = np.median(base_window_modIQ)
    base_Uci   = np.median(base_window_Uci)

    # Épaisseur robuste du bruit de fond (entre le 5e et 95e percentile)
    noise_thickness = np.percentile(base_window_Uci, 95) - np.percentile(base_window_Uci, 5)
    
    # Mesure des êxtremes dans la zone de défaut
    max_modIQ_zone = modIQ.iloc[zone_start:zone_end].max()
    max_Uci_zone = Uci.iloc[zone_start:zone_end].max()
    
    # Deltas absolus
    delta_modIQ = max_modIQ_zone - base_modIQ
    delta_Uci = max_Uci_zone - base_Uci
    
    # Protection division par zéro  
    EPSILON = 0.01
    base_modIQ_safe = max(abs(base_modIQ), EPSILON)
    base_Uci_safe = max(abs(base_Uci), EPSILON)

    pct_modIQ = (delta_modIQ / base_modIQ_safe) * 100.0
    pct_Uci = (delta_Uci / base_Uci_safe) * 100.0

    modIQ_is_real_peak = delta_modIQ > (noise_thickness) and pct_modIQ > 10.0
    Uci_is_real_peak = delta_Uci > (noise_thickness) and pct_Uci > 10.0

    if modIQ_is_real_peak or Uci_is_real_peak:
        return "Other;Peak modIQ or Uci"

    return None

def _check_wrong_alarm_anomalies(time, Uci, Ucr, modIQ):
    """
    [Sous-fonction] Vérifie le comportement "Wrong alarm" (Chute standard).
    """
    zero_idx = np.argmin(np.abs(time))
    dt_us = np.mean(np.diff(time))

    # Fenêtres de mesure
    pts_before_start = max(1, int(440.0 / dt_us))
    pts_before_end = max(1, int(100.0 / dt_us))
    pts_after = max(1, int(550.0 / dt_us))

    idx_before_start = max(0, zero_idx - pts_before_start)
    idx_before_end   = max(1, zero_idx - pts_before_end)

    idx_after_start  = zero_idx
    idx_after_end    = min(len(Uci), zero_idx + pts_after)

    if idx_after_start >= len(Uci) - 2:
        return None

    # Extraction des niveaux calmes juste avant la coupure
    ucr_before = np.median(Ucr.iloc[idx_before_start:idx_before_end])

    # Extraction des extrêmes atteints juste après la coupure
    uci_after = Uci.iloc[idx_after_start:idx_after_end].min()
    iq_after  = modIQ.iloc[idx_after_start:idx_after_end].min()
    ucr_after = Ucr.iloc[idx_after_start:idx_after_end].max()

    # Calcul du pourcentage de variations
    delta_pos_Ucr = ucr_after - ucr_before
    ucr_before_safe = max(abs(ucr_before), 0.001)
    pct_pos_Ucr = (delta_pos_Ucr / ucr_before_safe) * 100.0

    # Évaluation des conditions
    # Uci et modIQ chutent presque à 0
    is_uci_zero = uci_after < 0.5
    is_iq_zero = iq_after < 0.5

    # Ucr monte très vite
    is_ucr_spike = pct_pos_Ucr > 10.0

    # Verdict
    if is_uci_zero and is_iq_zero and is_ucr_spike:
        return "Wrong alarm"

    return None

def classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, beam_presence, t0_ampt_idx):
    """
    Classifie le défaut "Rég signal RF hors tolérance". 
    """
    # Temps entre chaque point du signal en microsecondes
    dt_us = np.mean(np.diff(time))
    dt_seconds = dt_us * 1e-6 # Conversion en secondes pour obtenir la fréquence en Hz et pas Mhz 

    # Détection d'un éventuel palier (escalier) avant le défaut
    end_stair_idx = detect_end_of_staircase(Ucav, t0_ampt_idx, dt_us)

    # Détection du décrochage (t0_zscore)
    if end_stair_idx is not None:
        # On cherche le décrochage uniquement après l'escalier
        print(f"end_stair_idx : {end_stair_idx} end_stair time : {time[end_stair_idx]}")
        t0_zscore_idx, _ = detect_t0_zscore(Ucav.iloc[end_stair_idx:])
        t0_zscore_idx += end_stair_idx
    else:
        # Recherche normal sur tout le signal
        t0_zscore_idx, _ = detect_t0_zscore(Ucav)

    # Affichage t0 zcore et ampt
    #print(f't0_zscore_idx : {t0_zscore_idx}, t0 zscore valeur : {time[t0_zscore_idx]}, t0_ampt_idx : {t0_ampt_idx}, t0 ampt valeur : {time[t0_ampt_idx]}')

    # Test des Oscillations
    oscillation_label = _check_instability_oscillations(
        Uci, modIQ, Ucav, dt_seconds, t0_zscore_idx, t0_ampt_idx, end_stair_idx
    )
    
    if oscillation_label: 
        return oscillation_label

    # Test des Quenchs
    # On définit la limite de début de recherche (après l'escalier si présent)
    start_limit = end_stair_idx if end_stair_idx is not None else 0
    
    start_idx_Uci_rise = detect_uci_rise_start(Uci, t0_ampt_idx, dt_us, start_limit_idx=start_limit)

    #Affichage début montée Uci
    #print(f'start_idx_Uci_rise : {start_idx_Uci_rise}, start_idx_Uci_rise val : {time[start_idx_Uci_rise]}')

    quench_label = _check_quench_signatures(
        time, Ucav, Uci, Ucr, t0_zscore_idx, start_idx_Uci_rise, end_stair_idx, t0_ampt_idx
    )
    
    if quench_label: 
        return quench_label

    # Test des anomalies de pics
    peaks_label = _check_amplitude_anomalies(
        time, Uci, Ucav, modIQ, beam_presence, t0_ampt_idx, t0_zscore_idx
    )
    if peaks_label:
        return peaks_label

    # Test des anomalies liées à la présence du faisceau
    beam_label = _check_beam_anomalies(beam_presence, t0_zscore_idx)
    if beam_label:
        return beam_label

    # Test des detuning lent
    detuning_label = _check_detuning_anomalies(
        time, Uci, Ucr, start_idx_Uci_rise, t0_ampt_idx
    )
    if detuning_label:
        return detuning_label

    # Test si des pics sur Uci et modIQ sont présent autour du moment du défaut
    other_peak = _check_other_peak(Uci, modIQ, t0_ampt_idx) 

    if other_peak:
        return other_peak

    # Test de chute standard (Wrong alarm)
    wrong_alarm_label = _check_wrong_alarm_anomalies(
        time, Uci, Ucr, modIQ
    )
    if wrong_alarm_label:
        return wrong_alarm_label
            
    # Si aucune condition n'est remplie
    return "Other"

def classify_RF_protections(time, Ucav, Ucat):
    """
    Classifie le défaut "Dép seuil de sécurité RF" en comparant la valeur de la tension (Ucav) par rapport au seuil maximal (Ucat). 
    """
    n_points = len(time)

    zero_idx = np.argmin(np.abs(time))
    
    # Fenêtre étroitre autour du défaut
    start_idx = max(0, zero_idx - 20)
    end_idx   = min(n_points, zero_idx + 20)
    
    # Calcul du ratio maximum
    Ucat_fraction = float(Ucat) / 100.0

    # On extrait la zone et on trouve la tension maximale atteinte
    search_window = Ucav.iloc[start_idx:end_idx]
    max_Ucav = search_window.max()

    max_ratio = max_Ucav / Ucat_fraction
    #print(f"Ratio maximum mesuré : {max_ratio:.2f} | Ucat_fraction : {Ucat_fraction}")

    # Classification
    threshold_ucav_max = CONFIG.get("thresholds", {}).get("rf_protection", {}).get("threshold_ucav_max", 8.5)
    
    if max_ratio > threshold_ucav_max:
        return "Ucav max"
    else:
        return "Pr max"

def classify_vacuum(pickup, void, zero_idx):
    """
    Classifie le défaut "Seuil de vide"
    Pour le moment, indique la valeur moyenne du courant pickup avant t0 et la valeur du vide à t0. 
    """
    pickup_win = pickup.iloc[0:zero_idx]
    mean_pickup = pickup_win.mean()

    vacuum_at_fault = void.iloc[zero_idx]

    text_pickup = f"Courant pickup moyen avant t0 : {mean_pickup:.3f} μA"
    text_vacuum = f"Vide à t0 : {vacuum_at_fault:.3e} mbar"

    return f";{text_pickup}, {text_vacuum}"

    '''
    threshold_pickup = 50
    if mean_pickup > threshold_pickup:
        return f"Ipu;{mean_pickup:.2f}uA,{mean_void:.2f}"
    else:
        return f"Leak/outgassing;{mean_pickup:.2f}uA,{mean_void:.2f}"
    '''

def classify_quench(Ucav, t0_ampt_idx):
    """
    Classifie le défaut "Claquage ou quench cavité" en mesurant le temps de chute 
    de la tension cavité (Ucav) de 100% à 50%.
    """
    cfg = CONFIG.get("thresholds", {}).get("quench", {})

    target_pct         = cfg.get("fall_target_pct", 50.0) / 100.0
    min_false_alarm_us = cfg.get("min_false_alarm_us", 1000.0)

    # Calcul du temps de chute
    result = calculate_fall_time(Ucav, t0_ampt_idx, target_pct=target_pct,z_threshold=100.0)
    
    if result is None:
        # print(f"Alerte : Le signal ne descend jamais à {target_pct*100}% dans ce fichier.")
        return f"Other;Incomplet (Pas de chute à {target_pct*100:.0f}%)"
        
    delta_t, u_max, t0_time, t_50 = result
    
    #print(f"U_max: {u_max:.2f} | t_start: {t0_time:.1f}µs | t_50: {t_50:.1f}µs | Chute en: {delta_t:.2f} µs")

    # Définition des seuils physiques
    dt_us = np.mean(np.diff(Ucav.index))

    threshold_breakdown = max(2.0, dt_us * 2)
    threshold_false_alarm = min_false_alarm_us

    # Classification
    fall_time_str = f";Temps de chute : {delta_t:.2f} us"

    if delta_t <= threshold_breakdown:
        return f"Breakdown{fall_time_str}"
    elif delta_t > threshold_false_alarm:
        return f"False alarm{fall_time_str}"
    else:
        # N'est pas sensé arriver
        return f"Quench{fall_time_str}"

def _check_filters(file_name, header, time, Ucav, zero_idx, ampt):
    """
    [Sous-fonction] Vérifie si les conditions d'exploitation du signal sont valides avant de procéderà son analyse complète. 
    S'appuie sur les paramètres définis dans le fichier de configuration.
    """
    cfg = CONFIG.get("filters", {})

    # Filtre Faisceau (BEAM) ou Boucle (LOOP) désactivés
    beam = header.get("BEAM").get("Valeur")
    loop = header.get("LOOP").get("Valeur")

    if cfg.get("check_beam", True) and beam == "NON":
        return f"Filtered;Beam NON"
        
    if cfg.get("check_loop", True) and loop == "OFF":
        return f"Filtered;Loop OFF"

    # Filtre sur KPI
    if cfg.get("check_kpi", True):
        kpi = float(header.get("KPI").get("Valeur"))
        kpi_threshold = cfg.get("min_kpi", 10.0)
        if kpi < kpi_threshold:
            return f"Filtered;KPI < {kpi_threshold}"

    # Filtre sur le Masque
    if cfg.get("check_mask", True):
        mask = header.get("MASK").get("Valeur")
        file_name_str = str(file_name)
        if "CMA" in file_name_str and mask not in ['0x007D', '0x007F']:
            return f"Filtered;Mask problem"
        if "CMB" in file_name_str and mask != "0x007F":
            return f"Filtered;Mask problem"

    #Filtre sur l'amplitude (AMPT)
    if cfg.get("check_ampt", True):
        ampt_threshold = cfg.get("max_ampt", 10.0)
        if ampt > ampt_threshold:
            return f"Filtered;AMPT > {ampt_threshold}%"
    
    # Filtre sur la tension cavité (Ucav)  
    if cfg.get("check_ucav_healthy", True):
        dt_us = np.mean(np.diff(time))
        ucav_min = cfg.get("ucav_min_healthy", 0.1)
        max_duration_us = cfg.get("ucav_low_max_duration_us", 200.0)

        Ucav_before_t0 = Ucav.iloc[:zero_idx]

        # Cas A : Tension presque nulle en moyenne avant le défaut
        if np.mean(Ucav_before_t0) < ucav_min:
            return f"Filtered;Tension moyenne < {ucav_min}"

        # Cas B : Tension qui tombe trop bas pendant un certain temps avant t0
        required_points = max(2, int(max_duration_us / dt_us))
        drop_mask = Ucav_before_t0 < ucav_min
        
        if drop_mask.rolling(window=required_points).sum().max() >= required_points:
            return f"Filtered;Tension < {ucav_min} avant t0"

def classify_signals(file_name, header, time, df_signals, df_defaults, filter=True):
    """
    Classifie un fichier post mortem à partir de signaux déjà chargés en mémoire.
    """
    # Identification du premier défaut déclenché
    faulty_rows = (df_defaults == 0).any(axis=1)
    if faulty_rows.any():
        first_fault_time = faulty_rows.idxmax()
        faults_at_time = df_defaults.loc[first_fault_time]
        detected_faults = faults_at_time[faults_at_time == 0].index.tolist()
        primary_fault = detected_faults[0]
    else:
        primary_fault = ""
    #print(f"Defaut : {detected_faults}")

    Ucav = df_signals.get("Ucav")
    zero_idx = np.argmin(np.abs(time))
    ampt = float(header.get("AMPT").get("Valeur"))

    # Vérification de l'état des filtres
    filter_result = _check_filters(file_name, header, time, Ucav, zero_idx, ampt)
    if filter and not filter_result is None: 
        return f"{primary_fault};{filter_result}"

    # Préparation du temps de départ
    t0_ampt_idx, _ = find_true_t0_with_ampt(Ucav,zero_idx, ampt)

    # Aiguillage vers les sous-classificateurs
    match primary_fault:
        case "Seuil pick-up" | "Coupure externe rapide" | "Absence autorisation RF":
            final_label = primary_fault
        
        case "Seuil de vide":
            pickup = df_signals.get("courant pickup")
            void = df_signals.get("vide")
            label = classify_vacuum(pickup, void, zero_idx)
            final_label = f"{primary_fault};{label}"
        
        case "Claquage ou quench cavité":
            label = classify_quench(Ucav, t0_ampt_idx)
            final_label = f"{primary_fault};{label}" 

        case "Dép seuil de sécurité RF":
            Ucat = header.get("UCAT").get("Valeur")
            label = classify_RF_protections(time, Ucav, Ucat)
            final_label = f"{primary_fault};{label}"

        case "Rég signal RF hors tolérance":
            Uci = df_signals.get("Uci")
            Ucr = df_signals.get("A Ucr") 
            modIQ = df_signals.get("IQ")
            beam_presence = df_signals.get("Presence faisceau")
            label = classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, beam_presence, t0_ampt_idx)
            final_label = f"{primary_fault};{label}"

        case _:
            final_label = "Pas de défaut"

    if not filter and filter_result is not None:
        filter_reason = filter_result.split(";")[-1]
        return f"{final_label}|{filter_reason}"

    return final_label


@chrono
def classify_file(file_name,filter=True):
    """
    Lit un fichier brut post-mortem depuis son chemin physique, puis le classifie.
    """
    with open(file_name, 'rb') as f:
        file_content = f.read()

    time, df_signals, df_defaults, _, header = Read_Signals(
        file_content, 
        compute_defauts=True, 
        compute_etats=True, 
        plot_signaux=False, 
        show_header=False
    )
     
    return classify_signals(file_name, header, time, df_signals, df_defaults, filter)
