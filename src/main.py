from classification import classify_file, classify_folder, classify_folder, classify_year_folder

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
    '''
    #file = '../../LLRF/2024/CMB/CMB07A/2024 10 02 CMB07-CAV1_0002' # quench aside
    file = '../../LLRF/2024/CMB/CMB02B/2024 11 08 CMB02-CAV2_0002' # quench
    #file = '../../LLRF/2024/CMA/CMA12/2024 11 08 CMA12_0006' # Power chain gain
    label = classify_file(file)
    print(f'Nom du fichier : {file} | Label :{label}')
    '''
    #classify_folder("../../LLRF/2024/CMA/CMA01")
    #classify_folder("../../LLRF/2024/CMB/CMB07A")
    #classify_cav_folder("../../LLRF/2024/CMB")
    classify_year_folder("../../LLRF/2024/")


#Seuil pick-up,
#Coupure externe rapide,    
#Absence autorisation RF,
#Seuil de vide,
#Claquage ou quench cavité,
#Dép seuil de sécurité RF,
#Rég signal RF hors tolérance