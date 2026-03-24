import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def save_nodes_influence(pop, iteration):
    folder = "exports/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    header = [
        "ID_Agent", "Pos_X", "Pos_Y",
        "Oeil_Avancer", "Oeil_Rotation",
        "Avancer_Rotation", "Rotation_Avancer",
        "Avancer_Oeil", "Rotation_Oeil"
    ]
    
    file_name = f"{folder}agents_save_{iteration}.csv"
    
    with open(file_name, mode="w", newline='', encoding="utf-8") as fichier_csv:
        writer = csv.writer(fichier_csv, delimiter=',')
        writer.writerow(header)
        
        max_depth = 5 
        
        for agent in pop:
            W = agent.sbn.weights
        
            # On initialise la matrice d'influence en float
            I = np.zeros_like(W, dtype=float)
            
            # Matrice qui va stocker les puissances successives
            W_power = np.eye(agent.sbn.num_nodes)
            
            # On utilise la profondeur fixe au lieu de num_nodes
            for k in range(1, max_depth + 1):
                W_power = np.matmul(W_power, W)
                I += W_power
            
            # Récupération des données spatiales et d'identification
            id_agent = agent.id
            pos_x = agent.x
            pos_y = agent.y
            
            # Extraction avec tes vrais index :
            # 0=Oeil, 1=Bouche, 2=Nageoire F (Avancer), 3=Nageoire R (Rotation)
            o_a = I[0, 2]  # Oeil -> Avancer
            o_r = I[0, 3]  # Oeil -> Rotation
            a_r = I[2, 3]  # Avancer -> Rotation
            r_a = I[3, 2]  # Rotation -> Avancer
            a_o = I[2, 0]  # Avancer -> Oeil
            r_o = I[3, 0]  # Rotation -> Oeil
            
            # Création et écriture de la ligne
            ligne_agent = [
                id_agent, pos_x, pos_y,
                o_a, o_r, a_r, r_a, a_o, r_o
            ]
            writer.writerow(ligne_agent)
            
    print(f"Sauvegarde des cerveaux générée à l'itération : {iteration} dans le fichier {file_name}")