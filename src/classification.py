import numpy as np
import pandas as pd
from PyPostMortem import Read_Signals
from utils import chrono
from signal_processing import (
    analyze_oscillation, 
    get_trend, 
    detect_t0_zscore, 
    detect_end_of_staircase
)

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

    end_stair = detect_end_of_staircase(Ucav,zero_idx, dt_microseconds)
    if end_stair is not None:
        print(f"End stair | idx : {end_stair}, valeur : {time[end_stair]}")
        #t0_idx = end_stair
        t0_idx, _ = detect_t0_zscore(Ucav.iloc[end_stair:])
        t0_idx += end_stair
    else:
        # t0_idx : index ou le signal Ucav comment à décrocher
        t0_idx, _ = detect_t0_zscore(Ucav)
    if t0_idx is None:
        # Si on ne trouve pas de t0, ça sert à rien de lancer la classification
        return "Other;Pas de chute de Ucav"
    if t0_idx == 0:
        return "Other;Problème t0"

    print(f't0_idx : {t0_idx}, t0 valeur : {time[t0_idx]}, index de 0 : {zero_idx}')
    t0_idx_Ucr, _ = detect_t0_zscore(Uci.iloc[end_stair:])

    # On isole la fenêtre d'analyse proche de t0 (avant la coupure)
    # Plage d'analyze pour la tendance global des signaux, et pour pour la tendance plus local
    WINDOW_POINTS_LARGE = 3500
    WINDOW_POINTS_SHORT = 700

    start_idx_large = max(0, t0_idx - WINDOW_POINTS_LARGE)
    start_idx_short = max(0, t0_idx - WINDOW_POINTS_SHORT)
    if end_stair is not None:
        start_idx_short = t0_idx_Ucr + end_stair
    else:
        start_idx_short = t0_idx_Ucr
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
        "Uci":  0.002,
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
    idx_base_start = max(0, zero_idx - 500)
    idx_base_end = max(1, zero_idx - 100)
    base_modIQ = np.median(modIQ.iloc[idx_base_start:idx_base_end])
    base_Uci   = np.median(Uci.iloc[idx_base_start:idx_base_end])
    base_Ucr   = np.median(Ucr.iloc[idx_base_start:idx_base_end])
    base_Ucav  = np.median(Ucav.iloc[idx_base_start:idx_base_end])
    #print(f'median modIQ : {base_modIQ}, median Uci : {base_Uci}, median Ucr : {base_Ucr}, median Ucav : {base_Ucav}')
    
    # Définition de la zone de recherche de pic (Autour de t0)
    # On cherche dans une petite fenêtre avant et après t0
    zone_start = zero_idx - 20 #max(0, min(t0_idx, zero_idx))
    zone_end   = zero_idx + 30
    
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
    t0_idx, t0_time = detect_t0_zscore(Ucav)
    t0_idx = t0_idx - 1
    t0_time = Ucav.index[t0_idx]

    u_max = Ucav.iloc[t0_idx]
    u_target = u_max * 0.5
    
    ucav_fall = Ucav.iloc[t0_idx:]
    below_50_points = ucav_fall[ucav_fall < u_target]
    
    if below_50_points.empty:
        return "Other;Incomplet (Pas de chute à 50%)"
        
    t_50 = below_50_points.index[0]
    delta_t = t_50 - t0_time
    
    print(f"U_max: {u_max:.2f} | t_start: {t0_time:.1f}µs | t_50: {t_50:.1f}µs | Chute en: {delta_t:.2f} µs")

    dt_microseconds = np.mean(np.diff(Ucav.index))
    THRESHOLD_BREAKDOWN = max(2.0, dt_microseconds * 1.05) # 2 µs
    THRESHOLD_FALSE_ALARM = 2000.0  # 2 ms (2000 µs)

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

    #Cas de tension presque nulle
    Ucav = df_signals.get("Ucav")
    zero_idx = np.argmin(np.abs(time))
    ucav_threshold = 0.1
    if np.mean(Ucav.iloc[0:zero_idx]) < ucav_threshold:
        return f'{detected_faults[0]};Other;Tension inférieur à {ucav_threshold}'

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
