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
    # Définition des seuils d'amplitude
    AMP_THRESHOLDS = {
        "Uci":  0.002,
        "modIQ": 0.1,
        "Ucav": 0.0001
    }
    
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
    	Uci.iloc[start_osci:end_osci],   dt_seconds, min_physical_amp=AMP_THRESHOLDS["Uci"]
    )
    osc_modIQ, f_modIQ = analyze_oscillation(
    	modIQ.iloc[start_osci:end_osci], dt_seconds, min_physical_amp=AMP_THRESHOLDS["modIQ"]
    )
    osc_Ucav, f_Ucav   = analyze_oscillation(
    	Ucav.iloc[start_osci:end_osci],  dt_seconds, min_physical_amp=AMP_THRESHOLDS["Ucav"]
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
            if f_Ucav > 100:
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
    target_pct = 0.01
    
    result = calculate_fall_time(Ucav, zero_idx, target_pct=target_pct)
    
    if result is None:
        # print(f"Alerte : Le signal ne descend jamais à {100 - target_pct * 100}% dans ce fichier.")
        return None

    delta_t, u_max, t0_time, t = result
    # print(f"U_max: {u_max:.2f} | t_start: {t0_time:.1f}µs | t: {t:.1f}µs | Chute en: {delta_t:.2f} µs")

    dt_us = np.mean(np.diff(time))
    threshold_breakdown = max(2.0, dt_us * 1.05) # ~11 µs
    threshold_false_alarm = 10000.0  # 10 ms
    
    if threshold_breakdown < delta_t < threshold_false_alarm:
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
    
def _check_amplitude_anomalies(time, Uci, Ucav, modIQ, t0_ampt_idx, t0_zscore_idx):
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
    THRESHOLD_NOISE_PCT = 10.00
    THRESHOLD_PEAK_PCT = 90.00

    # Un vrai pic doit dépasser significativement le bruit ambiant
    is_real_peak = delta_pos_Uci > (noise_thickness)

    # Sélection du pourcentage Uci pertinent (positif si vrai pic, sinon on garde le creux négatif)
    if pct_pos_Uci > THRESHOLD_NOISE_PCT and is_real_peak:
        pct_Uci = pct_pos_Uci
    else:
        pct_Uci = pct_neg_Uci
    
    # Tendance très courte sur Ucav au moment exact du décrochage (sécurisé contre les index négatifs)
    trend_start = max(0, t0_zscore_idx - 1)
    trend_end   = t0_zscore_idx + 5
    trend_Ucav_short = get_trend(time[trend_start:trend_end], Ucav.iloc[trend_start:trend_end], tolerance=1e-6)
    #print(f'pct_Uci : {pct_Uci}, is_real_peak : {is_real_peak}, trend_Ucav_short : {trend_Ucav_short}')
   
    if pct_modIQ > THRESHOLD_NOISE_PCT:

        # Power chain gain, gros pic modIQ, suivit d'une chute de Uci
        if pct_modIQ > THRESHOLD_PEAK_PCT and pct_Uci < -THRESHOLD_NOISE_PCT:
            return "Power chain gain"

        # Beam loss (Pic modIQ et Uci, chute de Ucav)
        elif pct_Uci > THRESHOLD_NOISE_PCT and is_real_peak and trend_Ucav_short == -1:
            return "Beam loss"

        # Beam loss mais montée de Ucav
        #elif pct_Uci > THRESHOLD_NOISE_PCT and trend_Ucav_short == 1:
            #return "Other,Faulty beam presence"
    
    return None

def classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, t0_ampt_idx):
    """
    Classifie le défaut "Rég signal RF hors tolérance". 
    """
    # Temps entre chaque point du signal en microsecondes
    dt_us = np.mean(np.diff(time))
    dt_seconds = dt_us * 1e-6 # Conversion en secondes pour obtenir la fréquence en Hz et pas Mhz 

    # Détection d'un éventuel palier (escalier) avant le défaut
    end_stair_idx = detect_end_of_staircase(Ucav,t0_ampt_idx, dt_us)

    # Détection du décrochage (t0_zscore)
    if end_stair_idx is not None:
        # On cherche le décrochage uniquement après l'escalier
        t0_zscore_idx, _ = detect_t0_zscore(Ucav.iloc[end_stair_idx:])
        t0_zscore_idx += end_stair_idx
    else:
        # Recherche normal sur tout le signal
        t0_zscore_idx, _ = detect_t0_zscore(Ucav)

    # Affichage t0 zcore et ampt
    print(f't0_zscore_idx : {t0_zscore_idx}, t0 zscore valeur : {time[t0_zscore_idx]}, t0_ampt_idx : {t0_ampt_idx}, t0 ampt valeur : {time[t0_ampt_idx]}')

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
        time, Uci, Ucav, modIQ, t0_ampt_idx, t0_zscore_idx
    )
    if peaks_label:
        return peaks_label
            
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
    threshold_ucav_max = 8.5 # Seuil empirique de distinction
    
    if max_ratio > threshold_ucav_max:
        return "Ucav max"
    else:
        return "Pr max"

def classify_void(pickup, void, t0_ampt_idx, threshold_pickup=50):
    """
    Classifie le défaut "Seuil de vide" en mesurant la quantité de courant pickup par rapport au seuil. 
    """
    pickup_win = pickup.iloc[0:t0_ampt_idx] #+ 38
    mean_pickup = pickup_win.mean() #+ 38
    void_win = void.iloc[0:t0_ampt_idx] * 10000000
    mean_void = void_win.mean()
    #print(f'moyenne pickup : {mean_pickup}')
    #print(f'moyenne void : {mean_void}')

    if mean_pickup > threshold_pickup:
        return f"Ipu;{mean_pickup:.2f}uA,{mean_void:.2f}"
    else:
        return f"Leak/outgassing;{mean_pickup:.2f}uA,{mean_void:.2f}"

def classify_quench(Ucav, t0_ampt_idx):
    """
    Classifie le défaut "Claquage ou quench cavité" en mesurant le temps de chute 
    de la tension cavité (Ucav) de 100% à 50%.
    """
    # Calcul du temps de chute
    result = calculate_fall_time(Ucav, t0_ampt_idx, target_pct=0.5,z_threshold=100.0)
    
    if result is None:
        #print("Alerte : Le signal ne descend jamais à 50% dans ce fichier.")
        return "Other;Incomplet (Pas de chute à 50%)"
        
    delta_t, u_max, t0_time, t_50 = result
    
    #print(f"U_max: {u_max:.2f} | t_start: {t0_time:.1f}µs | t_50: {t_50:.1f}µs | Chute en: {delta_t:.2f} µs")

    # Définition des seuils physiques
    dt_us = np.mean(np.diff(Ucav.index))

    threshold_breakdown = max(2.0, dt_us * 2.05) # environ 11 µs (en général)
    threshold_false_alarm = 1000.0  # 1 ms (1000 µs)

    # Classification
    fall_time_str = f";Temps de chute : {delta_t:.2f} us"

    if delta_t <= threshold_breakdown:
        return f"Breakdown{fall_time_str}"
    elif delta_t > threshold_false_alarm:
        return f"False alarm{fall_time_str}"
    else:
        # N'est pas sensé arriver
        return f"Quench{fall_time_str}"

@chrono
def classify_file(file_name):
    """
    Classifie le fichier
    """
    with open(file_name, 'rb') as f:
        file_content = f.read()

    parameters, time, df_signals, df_defaults, _, _ = Read_Signals(
        file_content, 
        compute_defauts=True, 
        compute_etats=False, 
        plot_signaux=False, 
        show_header=False
    )
     
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

    cfg = CONFIG.get("filters", {})

    # Filtre Faisceau (BEAM) ou Boucle (LOOP) désactivés
    beam = parameters.get("BEAM")
    loop = parameters.get("LOOP")

    if cfg.get("check_beam", True) and beam == "NON":
        return f"{primary_fault};Filtered;Beam NON"
        
    if cfg.get("check_loop", True) and loop == "OFF":
        return f"{primary_fault};Filtered;Loop OFF"

    # Filtre sur KPI
    if cfg.get("check_kpi", True):
        kpi = float(parameters.get("KPI", 0))
        kpi_threshold = cfg.get("min_kpi", 10.0)
        if kpi < kpi_threshold:
            return f"{primary_fault};Filtered;KPI < {kpi_threshold}"

    # Filtre sur le Masque
    mask = parameters.get("MASK")
    file_name_str = str(file_name)

    if cfg.get("check_mask", True):
        mask = parameters.get("MASK")
        file_name_str = str(file_name)
        if "CMA" in file_name_str and mask not in ['0x007D', '0x007F']:
            return f"{primary_fault};Filtered;Mask problem"
        if "CMB" in file_name_str and mask != "0x007F":
            return f"{primary_fault};Filtered;Mask problem"

    #Filtre sur l'amplitude (AMPT)
    ampt = float(parameters.get("AMPT", 3.0))
    if cfg.get("check_ampt", True):
        ampt_threshold = cfg.get("max_ampt", 10.0)
        if ampt > ampt_threshold:
            return f"{primary_fault};Filtered;AMPT > {ampt_threshold}%"
    
    # Filtre sur la tension cavité (Ucav)  
    Ucav = df_signals.get("Ucav")
    zero_idx = np.argmin(np.abs(time))
    
    if cfg.get("check_ucav_healthy", True):
        dt_us = np.mean(np.diff(time))
        ucav_min = cfg.get("ucav_min_healthy", 0.1)
        max_duration_us = cfg.get("ucav_low_max_duration_us", 200.0)

        Ucav_before_t0 = Ucav.iloc[:zero_idx]

        # Cas A : Tension presque nulle en moyenne avant le défaut
        if np.mean(Ucav_before_t0) < ucav_min:
            return f'{primary_fault};Filtered;Tension moyenne < {ucav_min}'

        # Cas B : Tension qui tombe trop bas pendant un certain temps avant t0
        required_points = max(2, int(max_duration_us / dt_us))
        drop_mask = Ucav_before_t0 < ucav_min
        
        if drop_mask.rolling(window=required_points).sum().max() >= required_points:
            return f"{primary_fault};Filtered;Tension < {ucav_min} avant t0" 

    # Préparation du temps de départ
    t0_ampt_idx, _ = find_true_t0_with_ampt(Ucav,zero_idx, ampt)

    # Aiguillage vers les sous-classificateurs
    match primary_fault:
        case "Seuil pick-up" | "Coupure externe rapide" | "Absence autorisation RF":
            return primary_fault
        
        case "Seuil de vide":
            pickup = df_signals.get("courant pickup")
            void = df_signals.get("vide")
            label = classify_void(pickup, void, t0_ampt_idx)
            return f"{primary_fault};{label}"
        
        case "Claquage ou quench cavité":
            label = classify_quench(Ucav,t0_ampt_idx)
            return f"{primary_fault};{label}" 

        case "Dép seuil de sécurité RF":
            Ucat = parameters.get("UCAT")
            label = classify_RF_protections(time, Ucav, Ucat)
            return f"{primary_fault};{label}"

        case "Rég signal RF hors tolérance":
            Uci = df_signals.get("Uci")
            Ucr = df_signals.get("A Ucr") 
            modIQ = df_signals.get("IQ")
            label = classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ, t0_ampt_idx)
            return f"{primary_fault};{label}"

        case _:
            return "Other"
