import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def vitesse_convergence(dossier_parent):
    conditions = ["Condition_Bornee", "Condition_Non_Bornee"]
    colonnes_cibles = [
        "Oeil_Avancer", "Oeil_Rotation", 
        "Avancer_Rotation", "Rotation_Avancer", 
        "Avancer_Oeil", "Rotation_Oeil"
    ]
    
    donnees_globales = []
    
    for condition in conditions:
        chemin_condition = os.path.join(dossier_parent, condition)
        dossiers_runs = glob.glob(os.path.join(chemin_condition, "Run_*"))
        
        for dossier_run in dossiers_runs:
            nom_run = os.path.basename(dossier_run)
            fichiers_csv = glob.glob(os.path.join(dossier_run, "exports", "*.csv"))
            
            for fichier in fichiers_csv:
                nom_fichier = os.path.basename(fichier)
                try:
                    iteration = int(nom_fichier.split('_')[-1].replace('.csv', ''))
                except ValueError: continue
                
                df = pd.read_csv(fichier)
                
                # On convertie la valeur des poids en -1/0/1 (pour avoir le même ordre de grandeur)
                df_phenotype = np.sign(df[colonnes_cibles])
                
                # On calcule la diversité de CHAQUE trait (l'écart-type)
                # Puis on fait la moyenne (.mean()) de ces écarts-types
                instabilite_globale = df_phenotype.std().mean()
                
                donnees_globales.append({
                    "Condition": condition,
                    "Seed": nom_run,
                    "Iteration": iteration,
                    "Instabilite_Globale": instabilite_globale
                })
                
    # Création du Super-Tableau
    df_super = pd.DataFrame(donnees_globales)
    df_super = df_super.sort_values(by="Iteration")
    
    return df_super

def draw_convergence_evolution(df_super, dossier_parent):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(
        data=df_super, 
        x="Iteration", 
        y="Instabilite_Globale", 
        hue="Condition", 
        linewidth=2.5,
        errorbar=None
    )
    
    plt.title("L'Horloge Évolutive : Vitesse de Convergence", fontsize=15, fontweight='bold')
    plt.xlabel("Itérations de la simulation")
    plt.ylabel("Instabilité Comportementale Globale")
    
    # Ligne de repère visuelle pour le "calme plat"
    plt.axhline(0.1, color='black', linestyle='--', alpha=0.5, label="Seuil de Stabilité (Convergence)")
    plt.legend()
    plt.tight_layout()
    
    dossier_save = os.path.join(dossier_parent, "Analyse_Convergence")
    os.makedirs(dossier_save, exist_ok=True)
    chemin_sauvegarde = os.path.join(dossier_save, "Vitesse_Convergence.png")
    
    plt.savefig(chemin_sauvegarde, dpi=150)
    plt.close()
    
    print(f"Fichier générée : {chemin_sauvegarde}")

if __name__ == "__main__":
    DOSSIER_RESULTS = "results"
    
    df_convergence = vitesse_convergence(DOSSIER_RESULTS)
    if not df_convergence.empty:
        draw_convergence_evolution(df_convergence, DOSSIER_RESULTS)
    else:
        print("Aucune donnée à analyser.")