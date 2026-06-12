import numpy as np
import pandas as pd
from PyPostMortem import Read_Signals
from utils import chrono
from signal_processing import (
    analyze_oscillation, 
    get_trend, 
    detect_t0_zscore, 
    detect_end_of_staircase,
    calculate_fall_time
)

def _check_instability_oscillations(Uci, modIQ, Ucav, dt_seconds, t0_idx, zero_idx, end_stair_idx):
    """
    [Sous-fonction] Cherche les fréquences de Taconis ou loop instability.
    """
    AMP_THRESHOLDS = {
        "Uci":  0.002,
        "modIQ": 0.1,
        "Ucav": 0.001 #0.0001
    }
    
    #Si t0 est supérieur au zéro temporel, il peut quand même y avoir une oscillation (par exemple le signal oscille depuis le début)
    if t0_idx > zero_idx:
        if end_stair_idx:
            #Si il y a un escalier dans le signal, on analayse à la fin de l'escalier
            start_osci = end_stair_idx
        else:
            #Sinon on cherche à nouveau t0 mais en étant moins tolérant sur les périodes de calme du signal
            start_osci, _ = detect_t0_zscore(Ucav, calm=False)
            print(f'start osci {start_osci} | val {Ucav.index[start_osci]}')
            if start_osci > zero_idx:
                start_osci = max(0, zero_idx - 10000)
        end_osci = zero_idx
    else:
        dt_microseconds = np.mean(np.diff(Ucav.index))
        min_duration_us_1 = 1000 
        required_points_1 = max(2, int(min_duration_us_1 / dt_microseconds))
        min_duration_us_2 = 500
        required_points_2 = max(2, int(min_duration_us_2 / dt_microseconds))
        if t0_idx > zero_idx - required_points_1 and t0_idx < zero_idx - required_points_2:
            start_osci = max(0, zero_idx - 10000)
            end_osci = zero_idx
        else:
            start_osci = t0_idx
            end_osci = zero_idx + 10

    osc_Uci, f_Uci = analyze_oscillation(
        Uci.iloc[start_osci:end_osci], dt_seconds, min_physical_amp=AMP_THRESHOLDS["Uci"]
    )
    osc_modIQ, f_modIQ = analyze_oscillation(
        modIQ.iloc[start_osci:end_osci], dt_seconds, min_physical_amp=AMP_THRESHOLDS["modIQ"]
    )
    osc_Ucav, f_Ucav = analyze_oscillation(
        Ucav.iloc[start_osci:end_osci], dt_seconds, min_physical_amp=AMP_THRESHOLDS["Ucav"]
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
            if abs(ratio - closest_integer) <= 0.2:
                is_coherent = True
        
        if is_coherent:
            if f_Ucav > 100:
                return "Loop instability"
            else:
                return "Taconis"
        else:
            return "Oscillation"
            
    return None

def _check_quench_signatures(time, Ucav, Uci, Ucr, t0_idx, t0_idx_Uci, end_stair, zero_idx):
    """
    [Sous-fonction] Analyse les tendances pour détecter un profil en V typique d'un Quench.
    """
    WINDOW_POINTS_LARGE = 3500
    WINDOW_POINTS_SHORT = 700

    start_idx_large = max(0, t0_idx - WINDOW_POINTS_LARGE)
    start_idx_short = max(0, t0_idx - WINDOW_POINTS_SHORT)
    
    if end_stair is not None:
        start_idx_short = t0_idx_Uci + end_stair
    else:
        start_idx_short = t0_idx_Uci
        
    end_idx = t0_idx
    if start_idx_short > end_idx:
        end_idx = zero_idx

    t_win_large = time[start_idx_large:zero_idx]
    Uci_win_large = Uci.iloc[start_idx_large:zero_idx]
    Ucr_win_large = Ucr.iloc[start_idx_large:zero_idx]

    if start_idx_short < end_idx:
        # Couper la fenêtre en deux
        start_idx_short = max(0, start_idx_short - 200)
        mid_window = start_idx_short + np.argmin(Ucr.iloc[start_idx_short:end_idx]) 
        
        if mid_window == start_idx_short:
            mid_window = end_idx - int(WINDOW_POINTS_SHORT * 0.5)
            
        end_idx_right = mid_window + np.argmax(Ucr.iloc[mid_window:end_idx])
        #end_idx_right = end_idx
        #print(f'Window start : {start_idx_short}, Window mid : {mid_window}, Window end : {end_idx}') 
        #print(f'Window start value short: {time[start_idx_short]}, Window start value large: {time[start_idx_large]}, Window mid value: {time[mid_window]}, Window end value: {time[end_idx]}')

        t_win_left = time[start_idx_short:mid_window]
        t_win_right = time[mid_window:end_idx_right]
        #t_win_right = time[mid_window:end_idx]
        
        Ucr_win_left = Ucr.iloc[start_idx_short:mid_window]
        Ucr_win_right = Ucr.iloc[mid_window:end_idx_right]
        #Ucr_win_right = Ucr.iloc[mid_window:end_idx]

        # Analyse des tendances
        trend_Uci = get_trend(t_win_large, Uci_win_large, tolerance=1e-5)
        trend_Ucr = get_trend(t_win_large, Ucr_win_large, tolerance=1e-6)
        trend_Ucr_left = get_trend(t_win_left, Ucr_win_left, tolerance=1e-6)
        trend_Ucr_right = get_trend(t_win_right, Ucr_win_right, tolerance=1e-6)

        # Calcul de la perte d'amplitude
        start_val_Ucr = Ucr.iloc[start_idx_short]
        min_val_Ucr = Ucr.iloc[mid_window]
        fall_depth_Ucr = start_val_Ucr - min_val_Ucr
        #print(f'fall_depth_Ucr : {fall_depth_Ucr}')
        
        THRESHOLD_QUENCH = 0.01 
        true_v_profile = (trend_Ucr_left == -1 and trend_Ucr_right == 1 and fall_depth_Ucr > THRESHOLD_QUENCH)

        quench = False
        target_pct = 0.01
        result = calculate_fall_time(Ucav, target_pct=target_pct)
    
        if result is None:
            print(f"Alerte : Le signal ne descend jamais à {100 - target_pct * 100}% dans ce fichier.")
            #return f"Other;Incomplet (Pas de chute à {100 - target_pct * 100}%)"
            return None

        delta_t, u_max, t0_time, t_95 = result
        
        print(f"U_max: {u_max:.2f} | t_start: {t0_time:.1f}µs | t_95: {t_95:.1f}µs | Chute en: {delta_t:.2f} µs")

        dt_microseconds = np.mean(np.diff(Ucav.index))
        THRESHOLD_BREAKDOWN = max(2.0, dt_microseconds * 1.05) # 2 µs
        THRESHOLD_FALSE_ALARM = 10000.0  # 1 ms (1000 µs)
        
        if delta_t > THRESHOLD_BREAKDOWN and delta_t < THRESHOLD_FALSE_ALARM:
            quench = True

        if trend_Uci == 1:
            #if true_v_profile or trend_Ucr == -1:
            if quench:
                return "Quench"
                #Ajouté une vérif sur le temps de la chute pour être sur.
            elif trend_Ucr == 1:
                return "Quench aside"
                
    return None
    
def _check_amplitude_anomalies(time, Uci, Ucav, modIQ, zero_idx, t0_idx):
    """
    [Sous-fonction] Analyse les pics anormaux par rapport au bruit de fond (Beam loss, Gain).
    """
    idx_base_start = max(0, zero_idx - 1000)
    idx_base_end = max(1, zero_idx - 100)
    
    base_modIQ = np.median(modIQ.iloc[idx_base_start:idx_base_end])
    base_Uci   = np.median(Uci.iloc[idx_base_start:idx_base_end])
    base_Ucav  = np.median(Ucav.iloc[idx_base_start:idx_base_end])
    
    zone_start = zero_idx - 20
    zone_end   = zero_idx + 30
    
    max_modIQ_zone = modIQ.iloc[zone_start:zone_end].max()
    max_Uci_zone = Uci.iloc[zone_start:zone_end].max()
    min_Uci_zone = Uci.iloc[zone_start:zone_end].min()
    
    delta_modIQ = max_modIQ_zone - base_modIQ
    delta_pos_Uci = max_Uci_zone - base_Uci
    delta_neg_Uci = min_Uci_zone - base_Uci
    
    EPSILON = 0.01
    base_modIQ_safe = max(abs(base_modIQ), EPSILON)
    base_Uci_safe = max(abs(base_Uci), EPSILON)

    pct_modIQ = (delta_modIQ / base_modIQ_safe) * 100.0
    pct_pos_Uci = (delta_pos_Uci / base_Uci_safe) * 100.0
    pct_neg_Uci = (delta_neg_Uci / base_Uci_safe) * 100.0
    
    #print(f'index start : {zone_start} | index end : {zone_end}, start time value : {time[zone_start]} | end time value : {time[zone_end]}')
    #print(f'Deltas absolus -> modIQ: {delta_modIQ:.4f} | Uci_pos: {delta_pos_Uci:.4f} | Uci_neg: {delta_neg_Uci:.4f}')
    #print(f'Variations (%) -> modIQ: {pct_modIQ:.1f}% | Uci_pos: {pct_pos_Uci:.1f}% | Uci_neg: {pct_neg_Uci:.4f}%')
    
    THRESHOLD_NOISE_PCT = 10.00
    THRESHOLD_PEAK_PCT = 90.00

    std_Uci_win = np.std(Uci.iloc[idx_base_start:idx_base_end])
    #print(f'std Uci window : {np.std(Uci.iloc[idx_base_start:idx_base_end])}')
    
    pct_Uci = pct_pos_Uci if pct_pos_Uci > THRESHOLD_NOISE_PCT and delta_pos_Uci > std_Uci_win * 3 else pct_neg_Uci
    trend_Ucav_short = get_trend(time[t0_idx:t0_idx+5], Ucav.iloc[t0_idx:t0_idx+5], tolerance=1e-6)
   
    if pct_modIQ > THRESHOLD_NOISE_PCT:
        if pct_modIQ > THRESHOLD_PEAK_PCT and pct_Uci < -THRESHOLD_NOISE_PCT:
            return "Power chain gain"
        elif pct_Uci > THRESHOLD_NOISE_PCT and delta_pos_Uci > std_Uci_win * 3 and trend_Ucav_short == -1:
            return "Beam loss"
        #elif pct_Uci > THRESHOLD_NOISE_PCT and trend_Ucav_short == 1:
            #return "Beam loss;Pic Ucav"
    
    return None

def classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ):
    """
    Classifie le défaut "Rég signal RF hors tolérance". 
    """
    # zero_idx : index de 0.0 dans time (ou la valeur la plus proche de 0.0)
    # Récupération de l'index de la valeur de temps la plus proche de 0.0
    zero_idx = np.argmin(np.abs(time))
    # Temps entre chaque point du signal en microsecondes
    dt_microseconds = np.mean(np.diff(time))

    end_stair_idx = detect_end_of_staircase(Ucav,zero_idx, dt_microseconds)
    if end_stair_idx is not None:
        print(f"End stair | idx : {end_stair_idx}, valeur : {time[end_stair_idx]}")
        #t0_idx = end_stair
        t0_idx, _ = detect_t0_zscore(Ucav.iloc[end_stair_idx:])
        t0_idx += end_stair_idx
    else:
        # t0_idx : index ou le signal Ucav comment à décrocher
        t0_idx, _ = detect_t0_zscore(Ucav)
    if t0_idx is None:
        return "Other;Pas de chute de Ucav"
    if t0_idx == 0:
        return "Other;Problème t0"

    print(f't0_idx : {t0_idx}, t0 valeur : {time[t0_idx]}, index de 0 : {zero_idx}')

    # Test des Oscillations
    dt_seconds = dt_microseconds * 1e-6 # Conversion en secondes pour obtenir la fréquence en Hz et pas Mhz 
    oscillation_label = _check_instability_oscillations(Uci, modIQ, Ucav, dt_seconds, t0_idx, zero_idx, end_stair_idx)
    
    if oscillation_label: 
        return oscillation_label

    # Test des Quenchs
    t0_idx_Uci, _ = detect_t0_zscore(Uci, calm=False)
    print(f't0_idx_Uci : {t0_idx_Uci}, t0_idx_Uci val : {time[t0_idx_Uci]}')

    quench_label = _check_quench_signatures(time, Ucav, Uci, Ucr, t0_idx, t0_idx_Uci, end_stair_idx, zero_idx)
    
    if quench_label: 
        return quench_label

    # Test des anomalies de pics
    peaks_label = _check_amplitude_anomalies(time, Uci, Ucav, modIQ, zero_idx, t0_idx)
    if peaks_label:
        return peaks_label
            
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
    #print(f"Index de t=0 : {zero_idx} | Ratio maximum mesuré : {max_ratio:.2f}")
    
    if max_ratio > 8.5:
        return "Ucav max"
    else:
        return "Pr max"

def classify_void(pickup, threshold_pickup=50):
    """
    Classifie le défaut "Seuil de vide" en mesurant la quantité de courant pickup par rapport au seuil. 
    """
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
    result = calculate_fall_time(Ucav, target_pct=0.5,z_threshold = 100)
    
    if result is None:
        print("Alerte : Le signal ne descend jamais à 50% dans ce fichier.")
        return "Other;Incomplet (Pas de chute à 50%)"
        
    delta_t, u_max, t0_time, t_50 = result
    
    print(f"U_max: {u_max:.2f} | t_start: {t0_time:.1f}µs | t_50: {t_50:.1f}µs | Chute en: {delta_t:.2f} µs")

    dt_microseconds = np.mean(np.diff(Ucav.index))
    THRESHOLD_BREAKDOWN = max(2.0, dt_microseconds * 2.05) # 2 µs
    THRESHOLD_FALSE_ALARM = 1000.0  # 1 ms (1000 µs)

    fall_time_str = f";Temps de chute : {delta_t:.2f} us"

    if delta_t <= THRESHOLD_BREAKDOWN:
        return "Breakdown" + fall_time_str
    elif delta_t > THRESHOLD_FALSE_ALARM:
        return "False alarm" + fall_time_str
    else:
        return "Quench" + fall_time_str

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
     
    faulty_rows = (df_defaults == 0).any(axis=1)
    if faulty_rows.any():
        first_fault_time = faulty_rows.idxmax()
        faults_at_time = df_defaults.loc[first_fault_time]
        detected_faults = faults_at_time[faults_at_time == 0].index.tolist()
    else:
        detected_faults = [""]
    #print(f"Defaut : {detected_faults}")

    # Faisceau désactivé ou boucle de régulation désactivé, Filtre beam = non ou loop = off
    beam = parameters.get("BEAM")
    loop = parameters.get("LOOP")
    if beam == "NON" or loop == "OFF":
        return f'{detected_faults[0]};Filtered;Loop NON ou Beam OFF'

    # Filtre sur Ucav
    Ucav = df_signals.get("Ucav")
    zero_idx = np.argmin(np.abs(time))
    dt_microseconds = np.mean(np.diff(time))
    Ucav_threshold = 0.1

    # Cas de tension qui tombe à moins de 0.1 avant le zéro temporel
    min_duration_us = 200 
    required_points = max(2, int(min_duration_us / dt_microseconds))
    Ucav_before_t0 = Ucav.iloc[:zero_idx]
    drop_mask = Ucav_before_t0 < Ucav_threshold
    if drop_mask.rolling(window=required_points).sum().max() >= required_points:
        return f"{detected_faults[0]};Filtered;Tension inférieur à {Ucav_threshold} avant t0"   

    # Cas de tension presque nulle
    if np.mean(Ucav.iloc[0:zero_idx]) < Ucav_threshold:
        return f'{detected_faults[0]};Filtered;Tension inférieur à {Ucav_threshold}'

    # Aiguillage selon le premier défaut trouvé
    match detected_faults[0]:
        case "Seuil pick-up":
            return detected_faults[0]
        case "Coupure externe rapide":
            return detected_faults[0]
        case "Absence autorisation RF":
            return detected_faults[0]
        case "Seuil de vide":
            pickup = df_signals.get("courant pickup")
            return detected_faults[0] + ";" + classify_void(pickup)
        case "Claquage ou quench cavité":
            return detected_faults[0] + ";" + classify_quench(Ucav)
        case "Dép seuil de sécurité RF":
            Ucat = parameters.get("UCAT")
            return detected_faults[0] + ";" + classify_RF_protections(time, Ucav, Ucat)
        case "Rég signal RF hors tolérance":
            Uci = df_signals.get("Uci")
            Ucr = df_signals.get("A Ucr") 
            modIQ = df_signals.get("IQ")
            return detected_faults[0] + ";"+ classify_Ecav_Instability(time, Uci, Ucr, Ucav, modIQ)
        case _:
            return "Other"
