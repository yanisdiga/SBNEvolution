import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_spatiale(dossier_parent):
    conditions = ["Condition_Bornee", "Condition_Non_Bornee"]
    donnees_globales = []
    
    for condition in conditions:
        chemin_condition = os.path.join(dossier_parent, condition)
        dossiers_runs = glob.glob(os.path.join(chemin_condition, "Run_*"))
        
        if not dossiers_runs:
            print(f"Aucune run trouvé dans {chemin_condition}")
            continue
            
        for dossier_run in dossiers_runs:
            nom_run = os.path.basename(dossier_run)
            fichiers_csv = glob.glob(os.path.join(dossier_run, "exports", "*.csv"))
            
            for fichier in fichiers_csv:
                nom_fichier = os.path.basename(fichier)
                try:
                    iteration = int(nom_fichier.split('_')[-1].replace('.csv', ''))
                except ValueError: continue
                
                # Ouverture d'un monde unique
                df = pd.read_csv(fichier)
                if len(df) < 6: # Sécurité si l'espèce est presque éteinte
                    continue
                    
                coordonnees = df[['Pos_X', 'Pos_Y']].values
                
                # Analyse KNN (Éparpillement) 
                knn = NearestNeighbors(n_neighbors=10)
                knn.fit(coordonnees)
                distances, _ = knn.kneighbors(coordonnees)
                score_eparpillement = np.mean(distances[:, 1:])
                
                # Analyse DBSCAN (Tribus)
                dbscan = DBSCAN(eps=50, min_samples=10)
                labels = dbscan.fit_predict(coordonnees)
                
                nb_tribus = len(set(labels)) - (1 if -1 in labels else 0)
                pourcentage_isoles = (list(labels).count(-1) / len(labels)) * 100
                
                # Sauvegarde de la métrique pure
                donnees_globales.append({
                    "Condition": condition,
                    "Seed": nom_run,
                    "Iteration": iteration,
                    "Score_Eparpillement": score_eparpillement,
                    "Nb_Tribus": nb_tribus,
                    "Pourcentage_Isoles": pourcentage_isoles
                })
                
    # Création du Super-Tableau
    df_super = pd.DataFrame(donnees_globales)
    df_super = df_super.sort_values(by="Iteration")
    
    return df_super

def draw_spatial_grah(df_super, dossier_parent):
    sns.set_theme(style="whitegrid") 
    dossier_save = os.path.join(dossier_parent, "Analyse_Spatiale_Macro")
    os.makedirs(dossier_save, exist_ok=True)
    
    # 1. Graphique du Score d'Éparpillement
    plt.figure(figsize=(12, 6))
    # Ajout de errorbar=None pour retirer la zone colorée (l'intervalle de confiance)
    sns.lineplot(data=df_super, x="Iteration", y="Score_Eparpillement", hue="Condition", linewidth=2, errorbar=None)
    plt.title("Dynamique d'Éparpillement Spatial (Plus c'est haut, plus ils s'évitent)", fontsize=14, fontweight='bold')
    plt.ylabel("Distance moyenne aux voisins (Pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_save, "Macro_Eparpillement.png"), dpi=150)
    plt.close()

    # 2. Graphique du Nombre de Tribus
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_super, x="Iteration", y="Nb_Tribus", hue="Condition", linewidth=2, errorbar=None)
    plt.title("Formation de Tribus Distinctes (DBSCAN Clusters)", fontsize=14, fontweight='bold')
    plt.ylabel("Nombre moyen de groupes denses")
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_save, "Macro_Nb_Tribus.png"), dpi=150)
    plt.close()

    # 3. Pourcentage d'Isolés (Le bruit)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_super, x="Iteration", y="Pourcentage_Isoles", hue="Condition", linewidth=2, errorbar=None)
    plt.title("Taux d'Individus Isolés (L'errance spatiale)", fontsize=14, fontweight='bold')
    plt.ylabel("% d'agents hors-tribus")
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_save, "Macro_Isoles.png"), dpi=150)
    plt.close()
    
    print(f"Comparatif générée dans le dossier : {dossier_save}")

if __name__ == "__main__":
    # Assure-toi que tes 8 runs sont bien dans results/Condition_Bornee/Run_X etc.
    DOSSIER_RESULTS = "results"
    
    df_macro_spatial = analyse_spatiale(DOSSIER_RESULTS)
    
    if not df_macro_spatial.empty:
        # On sauvegarde aussi le fichier Excel/CSV des datas pour tes archives
        df_macro_spatial.to_csv(os.path.join(DOSSIER_RESULTS, "Base_Spatiale_Multi_Runs.csv"), index=False)
        draw_spatial_grah(df_macro_spatial, DOSSIER_RESULTS)
    else:
        print("Aucune donnée à analyser.")