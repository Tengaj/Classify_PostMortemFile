# Date: 22/04/2026
# Nom: PyPostMortem
# Description: lecture des fichiers postmortem
# Version: 2.1
 
class LengthError(Exception):
    pass

class HeaderACQError(Exception):
    pass

class ChunkError(Exception):
    pass

def Read_Header(filename,show_header=False):
    import re

    def extraire_cle_valeur_unite(chaine):
        # Séparer la clé et l'ensemble (valeur + unité) en fonction du caractère "="
        if "=" in chaine:
            cle, valeur_unite = chaine.split("=", 1)  # Diviser en 2 au niveau du premier "="
            cle = cle.strip()  # Nettoyer les espaces autour de la clé
            
            # Expression régulière pour capturer la valeur et l'unité optionnelle
            match = re.match(r'(.+?)\s*(?:\((.*?)\))?$', valeur_unite.strip())
            if match:
                valeur = match.group(1).strip()  # Capturer la valeur
                unite = match.group(2).strip() if match.group(2) else None  # Capturer l'unité (si elle existe)
                return cle, valeur, unite
            else:
                # Si le format n'est pas respecté pour la partie "valeur (unité)"
                return cle, None, None
        else:
            # Si le format "clé=valeur" n'est pas respecté
            return None, None, None

    def read_head(filename):
        Header_list=[]
        # Créer un dictionnaire à partir des résultats
        Header = {}
        with open(filename,'rb') as file:
            while True:
                line = file.readline()
                if not line:
                    break

                line = line.decode('utf-8').strip()
                # if show_header:
                #     print(line)

                if line == "END":
                    break

                Header_list.append(line)

            for array in Header_list:
                cle, valeur, unite = extraire_cle_valeur_unite(array)
                if cle:
                    # Ajouter au dictionnaire avec la clé et un sous-dictionnaire pour "Valeur" et "Unité"
                    Header[cle] = {
                        "Valeur": valeur,
                        "Unité": unite
                    }
            del Header_list

        return Header

    Header=read_head(filename)
    if Header != {}:
        return Header
    else:
        return None
    # Attention : POSTROW est en fait le nombre de ligne AVANT t=0 et non après, ce n'est pas égal au postrow de l'interface post-mortem (ihm) (un calcul est effectué : postrow_fichier = nrow - postrow_ihm)


def Read_Signals(file_content,compute_defauts=False,compute_etats=False,plot_signaux=False,show_header=False,etats_columns=None):
    import pandas as pd
    import numpy as np
    import re
    import math
    import io

    def extraire_cle_valeur_unite(chaine):
        if "=" in chaine:
            cle, valeur_unite = chaine.split("=", 1)
            cle = cle.strip()
            match = re.match(r'(.+?)\s*(?:\((.*?)\))?$', valeur_unite.strip())
            if match:
                valeur = match.group(1).strip()
                unite = match.group(2).strip() if match.group(2) else None
                return cle, valeur, unite
            else:
                return cle, None, None
        else:
            return None, None, None

    def read_head(file_content):
        Header_list=[]
        file = io.BytesIO(file_content)
        end_found = False
        while True:
            line = file.readline()
            if not line:
                break

            line = line.decode('utf-8').strip()
            # if show_header:
            #     print(line)

            if line == "END":
                end_found = True
                break

            Header_list.append(line)

        if not end_found:
            raise HeaderACQError("Header END introuvable")
        
            # Créer un dictionnaire à partir des résultats
        Header = {}

        for array in Header_list:
            cle, valeur, unite = extraire_cle_valeur_unite(array)
            if cle:
                # Ajouter au dictionnaire avec la clé et un sous-dictionnaire pour "Valeur" et "Unité"
                Header[cle] = {
                    "Valeur": valeur,
                    "Unité": unite
                }
        del Header_list

        values= []
        if Header["FORMAT"]["Valeur"]=="BINAIRE":
            try:
                raw_data = file.read()
                if len(raw_data) % 2 != 0:
                    raise ChunkError

                values = np.frombuffer(raw_data, dtype='>i2')
                if values.size % 16 != 0:
                    raise ChunkError

                values = values.reshape(-1, 16)
            except ValueError as exc:
                raise ChunkError from exc
            
        elif Header["FORMAT"]["Valeur"]=="ASCII":
            while True:
                chunk = file.readline()
                try:
                    chunk = chunk.decode("utf-8")
                    chunk = chunk.split()
                    chunk = [int(valeur, 16) for valeur in chunk]
                    if not chunk:
                        break
                    else:
                        values.append(chunk)
                except (UnicodeDecodeError, ValueError) as exc:
                    raise ChunkError from exc
        df = pd.DataFrame(values)
        df.columns = [Header["ACQ"+str(i)]["Valeur"] for i in range(1,17)]

        return Header,df

    def calculs_defauts_et_etats(time,df,compute_defauts,compute_etats,etats_columns=None):
        etats = ["Fsamp"                                     ,
                    "Switch RF ON"                           ,
                    "Multipac"                               ,
                    "VIR"                                   ,
                    "TSR"                                   ,
                    "Temp RF int"                           ,
                    "Temp RF ext"                           ,
                    "Switch rap"                            ,
                    "Alim"                                  ,
                    "Tension"                               ,
                    "Polarisée"                             ,
                    "Présence RF cav"                       ,
                    "Seuil pick-up"                        ,
                    "RF hors spéc"                          ,
                    "Ucr"                                   ,
                    "FPGA prêt"                                    ]

        if compute_defauts:
            defauts = [ "Seuil pick-up"                      ,     
                        "Coupure externe rapide"              ,      
                        "Absence autorisation RF"              ,   
                        "Seuil de vide"                        ,    
                        "Claquage ou quench cavité"            , 
                        "Dép seuil de sécurité RF"              ,   
                        "Rég signal RF hors tolérance"              ]
            defauts_values = df.iloc[:,0].astype(np.int16).to_numpy().astype(np.uint16)
            defauts_bits = ((defauts_values[:, None] >> np.arange(7)) & 1).astype(np.int8)
            df_defaut = pd.DataFrame(defauts_bits, columns=defauts)

        if compute_etats:
            etats_to_compute = etats
            if etats_columns:
                etats_to_compute = [etat for etat in etats_columns if etat in etats]
                if len(etats_to_compute) == 0:
                    etats_to_compute = etats

            states_values = df.iloc[:,1].astype(np.int16).to_numpy().astype(np.uint16)
            states_bits = ((states_values[:, None] >> np.arange(16)) & 1).astype(np.int8)
            df_states = pd.DataFrame({
                etat: states_bits[:, etats.index(etat)]
                for etat in etats_to_compute
            })

        df_signaux = df.iloc[:,2:]

        if compute_defauts and not compute_etats:
            if (NROW != len(df_signaux)) & (NROW != len(df_defaut)):
                raise LengthError

            df_signaux.index = time
            df_defaut.index = time
            return [df_signaux,df_defaut,None]
        elif compute_etats and not compute_defauts:
            if (NROW != len(df_signaux)) & (NROW != len(df_states)):
                raise LengthError

            df_signaux.index=time
            df_states.index=time
            return [df_signaux,None,df_states]
        elif compute_defauts and compute_etats:
            if (NROW != len(df_signaux)) & (NROW != len(df_defaut)) & (NROW != len(df_states)):
                raise LengthError

            df_signaux.index=time
            df_defaut.index=time
            df_states.index=time

            return [df_signaux, df_defaut, df_states]
        else:
            if NROW != len(df_signaux):
                raise LengthError

            df_signaux.index=time
            return [df_signaux,None,None]

    Header,df=read_head(file_content)

    NDEC = int(Header["NDEC"]["Valeur"])
    NROW = int(Header["NROW"]["Valeur"])
    POSTROW = int(Header["POSTROW"]["Valeur"])
    DATE = Header["DATE"]["Valeur"]
    ALM = Header["ALM"]["Valeur"]
    DBNAME = Header["DBNAME"]["Valeur"]
    KPI = Header["KPI"]["Valeur"]
    KII = Header["KII"]["Valeur"]
    KPUCI = Header["KPUCI"]["Valeur"]
    KIUCI = Header["KIUCI"]["Valeur"]
    AMPT = Header["AMPT"]["Valeur"]
    PHIT=Header["PHIT"]["Valeur"]
    BEAM=Header["BEAM"]["Valeur"]
    typeCavite = Header["CAVITE"]["Valeur"]
    LOOP = Header["LOOP"]["Valeur"]
    MASK = Header["MASK"]["Valeur"]
    UCAT = Header["UCAT"]["Valeur"]

    parameters={
        "NDEC":NDEC,
        "NROW":NROW,
        "POSTROW":POSTROW,
        "DATE":DATE,
        "ALM":ALM,
        "DBNAME":DBNAME,
        "KPI":KPI,
        "KII":KII,
        "KPUCI":KPUCI,
        "KIUCI":KIUCI,
        "AMPT":AMPT,
        "PHIT":PHIT,
        "BEAM":BEAM,
        "CAVITE":typeCavite,
        "LOOP":LOOP,
        "MASK":MASK,
        "UCAT":UCAT,
    }

    # Attention : POSTROW fait référence au nombre de ligne AVANT t=0 et non après, ce n'est pas égal au postrow de l'interface post-mortem (ihm) (un calcul est effectué : postrow_fichier = nrow - postrow_ihm)

    def calculBaseTemps(NDEC,NROW,POSTROW):
        b_t = 4/70.442*NDEC       # pas d'échantillonage en µs
        tf  = (NROW-POSTROW)*b_t  # Temps post triger
        ti  = -(POSTROW-1)*b_t    # Temps initial
        time=np.linspace(ti,tf,NROW) #  Vecteur temps à retourner
        return time

    time = calculBaseTemps(NDEC,NROW,POSTROW)

    def GetCaviteCoef(caviteNom):
        coefUci_cavs_base=2.7
        match caviteNom:
            case "RFQ_MAITRE":
                coefUcr=11.8
                coefUcav=165
                coefUci=111.8

            case "SLV2":
                coefUcr=11.6
                coefUcav=48.4
                coefUci=1

            case "SLV3":
                coefUcr=11.5
                coefUcav=72.7
                coefUci=1

            case "SLV4":
                coefUcr=12
                coefUcav=48
                coefUci=1

            case "REGROUPEUR_LME":
                coefUcr=1
                coefUcav=262.6
                coefUci=1

            case "CAVITE_A":
                coefUcr=0
                coefUcav=9
                coefUci=coefUci_cavs_base

            case "CAVITE_B":
                coefUcr=0
                coefUcav=8.7
                coefUci=coefUci_cavs_base+18

            case "REGROUPEUR_LHE":
                coefUcr=0
                coefUcav=100
                coefUci=100
        
        if math.isnan(coefUcav) or math.isnan(coefUci):
            coefUcav=100
            coefUci=100
        return coefUcr,coefUcav,coefUci

    if typeCavite == "RFQ_ESCLAVE":
        typeCavite = Header["DBNAME"]["Valeur"].split("-")[1].split(":")[0]

    coefUcr, coefUcav, coefUci  = GetCaviteCoef(typeCavite)
    if typeCavite == "CAVITE_A":
        if int(DBNAME.split("-")[1][-2:]) < 7:
            pass
        elif int(DBNAME.split("-")[1][-2:]) < 9:
            coefUci  = coefUci + 6
        else:
            coefUci  = coefUci + 12
            
    df, df_defaut, df_states = calculs_defauts_et_etats(time,df,compute_defauts,compute_etats,etats_columns)

    df_signaux = df
    del df
    df_signaux.columns = [f"{col} binary" for col in df_signaux.columns]

    if "A Ucr binary" in df_signaux.columns:
        ADO1 = float(Header["ADO1"]["Valeur"])
        ADA1 = float(Header["ADA1"]["Valeur"])
        # print(df_signaux._is_copy)
        df_signaux.loc[:,"A Ucr"] = 10**(((ADO1+(df_signaux["A Ucr binary"]*2.5/4095)/ADA1)+coefUcr)/10)

    if "A Uamp binary" in df_signaux.columns:
        ADO2 = float(Header["ADO2"]["Valeur"])
        ADA2 = float(Header["ADA2"]["Valeur"])
        # print(ADO2,ADA2)
        df_signaux.loc[:,"A Uamp"] = ADO2+(df_signaux["A Uamp binary"]*2.5/4095)/ADA2

    if "I modulateur binary" in df_signaux.columns:
        # df_signaux["I modulateur binary"]=df_signaux["I modulateur"]
        df_signaux.loc[:,"I modulateur"]= df_signaux["I modulateur binary"]/32767*300

    if "Q modulateur binary" in df_signaux.columns:
        # df_signaux["Q modulateur binary"]=df_signaux["Q modulateur"]
        df_signaux.loc[:,"Q modulateur"]= df_signaux["Q modulateur binary"]/32767*300

    if "vide binary" in df_signaux.columns:
        #v1      =
        df_signaux.loc[:,"vide"]= 1.332*10**(((df_signaux["vide binary"]*10.0)/4095)/0.6-12)

    if "courant pickup binary" in df_signaux.columns:
        df_signaux.loc[:,"courant pickup"]= df_signaux["courant pickup binary"]*0.0245168-37.5008

    if "T SP2RF interne binary" in df_signaux.columns:
        df_signaux.loc[:,"T SP2RF interne"]= (df_signaux["T SP2RF interne binary"]*256.0)/(4095-128)

    if "T SP2RF externe binary" in df_signaux.columns:
        df_signaux.loc[:,"T SP2RF externe"]= (df_signaux["T SP2RF externe binary"]*256.0)/(4095-128)

    if "I Ucav binary" in df_signaux.columns and "Q Ucav binary" in df_signaux.columns:
        df_signaux.loc[:,"Ucav"]= np.hypot(df_signaux["I Ucav binary"],df_signaux["Q Ucav binary"]) *coefUcav/32767

    if "I Uci binary" in df_signaux.columns and "Q Uci binary"  in df_signaux.columns:
            df_signaux.loc[:,"Uci"] = coefUci * ((df_signaux["I Uci binary"] / 32767.0)**2 + (df_signaux["Q Uci binary"] / 32767.0)**2)
            

    if "Q Ucav binary" in df_signaux.columns and "I Ucav binary" in df_signaux.columns:
        # df_signaux["PhaseCav"]=  df_signaux.apply(lambda row: math.atan2(row['Q Ucav'],row['I Ucav']), axis=1)
        # # Si tu veux convertir l'angle de radians à degrés
        # df_signaux['PhaseCav'] = df_signaux['PhaseCav'].apply(math.degrees)
        df_signaux.loc[:,'PhaseCav'] = (np.degrees(np.arctan2(df_signaux['Q Ucav binary'], df_signaux['I Ucav binary'])) - float(Header["UCAR"]["Valeur"]) + 180) % 360 - 180

    if "Q Uci binary" in df_signaux.columns and "I Uci binary" in df_signaux.columns:
        # df_signaux["PhaseUci"]= math.atan2(df_signaux["I Uci"], df_signaux["Q Uci"])*180/np.pi
        # df_signaux["PhaseUci"]=  df_signaux.apply(lambda row: math.atan2(row['Q Uci'],row['I Uci']), axis=1)
        # # Si on veux convertir l'angle de radians à degrés
        # df_signaux['PhaseUci'] = df_signaux['PhaseUci'].apply(math.degrees)

        df_signaux.loc[:,'PhaseUci'] = (np.degrees(np.arctan2(df_signaux['Q Uci binary'], df_signaux['I Uci binary'])) - float(Header["UCIR"]["Valeur"]) + 180) % 360 - 180

    # df_signaux["I Uci"]=df_signaux["I Uci"]/10.0

    if "PhaseCav" in df_signaux.columns and "PhaseUci" in df_signaux.columns:
        df_signaux.loc[:,"DPCI"]= (df_signaux["PhaseCav"] - df_signaux["PhaseUci"] - ((float(Header["DPOF"]["Valeur"])+180)%360 -180) + 180) % 360 -180

    if "I modulateur binary" in df_signaux.columns and "Q modulateur binary" in df_signaux.columns:
        df_signaux.loc[:,"IQ"]= np.hypot(df_signaux["I modulateur binary"],df_signaux["Q modulateur binary"]) /32767*300

    # if "IQ" in df_signaux.columns:
    #     df_signaux["MODA"]= 20*np.log10(df_signaux["IQ"])-57.75

    # if "Q modulateur binary" and "I modulateur binary" and "IQ binary" in df_signaux.columns:
    if "Q modulateur binary" in df_signaux.columns and "I modulateur binary" in df_signaux.columns:

        #df_signaux["MODP"]= np.sin(df_signaux["Q modulateur binary"]) * (180/np.pi) *(df_signaux["I modulateur binary"]/df_signaux["IQ"]).apply(cmath.acos)
        # df_signaux["MODP"]= np.arctan(df_signaux["Q modulateur"]/(df_signaux["I modulateur"]+1e-9))
        # df_signaux["MODP2"]= np.angle([df_signaux["I modulateur binary"]+ df_signaux["Q modulateur binary"]*1j])
        df_signaux.loc[:,"MODP"]= np.degrees(np.arctan2(df_signaux["Q modulateur binary"],(df_signaux["I modulateur binary"])))

    df_defaut = df_defaut.astype(bool)
    df_states = df_states.astype(bool)
    return parameters,time, df_signaux, df_defaut, df_states,Header