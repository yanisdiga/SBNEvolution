import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Fonction permettant d'afficher le graphique du réseaux de neuronnes d'un agent
def show_sbn_graph(agent_id, sbn):
    plt.figure(figsize=(18, 8))
    plt.suptitle(f"Cerveau de l'agent {agent_id} (Noeuds: {sbn.num_nodes})", fontsize=16)
    
    # Création du graphe orienté
    G = nx.DiGraph()
    
    # Dictionnaire de traduction
    name_nodes = {0: "E", 1: "M", 2: "F", 3: "R", 4: "R1", 5: "R2", 6: "R3", 7: "R4"}
    
    # On ajoute des noms génériques (N8, N9...) si le réseau a muté et grandi
    for i in range(8, sbn.num_nodes):
        name_nodes[i] = f"N{i}"

    # On crée chaque noeud de haut en bas en fonction de la fonction (pour plus de lisibilité)
    for id in sbn.true_ids:
        name = name_nodes.get(id, f"N{id}")
        if id in [0, 1]:         # Oeil, Bouche
            layer = 3           # Couche supérieure
        elif id in [4, 5, 6]:    # R1, R2, R3
            layer = 1           # Couche inférieure
        elif id in [3, 7]:       # R, R4
            layer = 0           # Couche inférieure
        else:                   # F et les nouveaux noeuds (N8, N9...)
            layer = 2           # Couche intermédiaire
            
        G.add_node(name, layer=layer)

    # On parcours de la matrice de poids
    edges_green = []
    edges_red = []
    
    for i in range(sbn.num_nodes):
        for j in range(sbn.num_nodes):
            poids = sbn.weights[i, j]
            if poids != 0:
                id_depart = sbn.true_ids[i]
                id_arrivee = sbn.true_ids[j]
                
                # On génère le nom dynamiquement s'il est nouveau (>= 8)
                nom_depart = name_nodes.get(id_depart, f"N{id_depart}")
                nom_arrivee = name_nodes.get(id_arrivee, f"N{id_arrivee}")
                
                # On ajoute la connexion avec son poids
                G.add_edge(nom_depart, nom_arrivee, weight=poids)
                
                # On trie pour l'affichage des couleurs
                if poids > 0:
                    edges_green.append((nom_depart,nom_arrivee))
                else:
                    edges_red.append((nom_depart, nom_arrivee))

    # align="horizontal" force les couches à se placer sur des lignes horizontales
    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")

    noeuds_bas = ["R", "R1", "R2", "R3", "R4"]
    # On récupère toutes les coordonnées X de la couche du bas et on les trie (pour ne pas avoir R, R2, R1, R4, R3 mais bien R,R1,...,R4)
    x_coords_bas = sorted([pos[n][0] for n in noeuds_bas if n in pos])
    
    # On réassigne ces coordonnées X triées à nos noeuds dans le bon ordre
    current_idx = 0
    for noeud in noeuds_bas:
        if noeud in pos:
            pos[noeud] = (x_coords_bas[current_idx], pos[noeud][1])
            current_idx += 1
    
    # 2. Aligner E, F et R verticalement sur la gauche
    # On prend la coordonnée X la plus à gauche (qui appartient maintenant à R)
    x_gauche = pos["R"][0]
    pos["E"] = (x_gauche, pos["E"][1])
    pos["F"] = (x_gauche, pos["F"][1])
    
    # Calcul dynamique des tailles de noeud, arrête et texte
    # Facteur d'échelle : 1.0 s'il y a 8 noeuds, 0.5 s'il y en a 16, etc.
    facteur_echelle = 8 / max(8, sbn.num_nodes)
    
    # On définit des limites minimales pour que ça reste visible à l'écran
    taille_noeud = max(500, int(2000 * facteur_echelle))
    taille_police = max(6, int(10 * facteur_echelle))
    taille_fleche = max(10, int(20 * facteur_echelle))
    
    # 1. Dessin des noeuds avec la taille dynamique
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=taille_noeud)
    nx.draw_networkx_labels(G, pos, font_size=taille_police, font_weight="bold")
    
    # 2. Dessin des arêtes (flèches)
    nx.draw_networkx_edges(G, pos, edgelist=edges_green, edge_color='green', 
                           arrows=True, arrowsize=taille_fleche, width=2,
                           node_size=taille_noeud)
                           
    nx.draw_networkx_edges(G, pos, edgelist=edges_red, edge_color='red', 
                           arrows=True, arrowsize=taille_fleche, width=2,
                           node_size=taille_noeud)
    
    # On récupère les poids de toutes les arêtes
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    # On dessine toutes les étiquettes manuellement
    for (u, v), weight in edge_labels.items():
        texte_poids = f"{weight:.2f}"
        
        if u == v:
            # Boucle (Self-loop) : on décale le texte au-dessus du noeud
            x, y = pos[u]
            plt.text(x, y + 0.15, texte_poids, fontsize=8, color='black', 
                     ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        else:
            # Connexion normale : on calcule le centre géométrique parfait
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            
            # On décale le texte en fonction de l'ordre alphabétique (cela empêche les flèches "Aller" et les flèches "Retour" de superposer leurs étiquettes)
            if str(u) < str(v):
                x_mid += 0.02
                y_mid += 0.04
            else:
                x_mid -= 0.02
                y_mid -= 0.04
                
            plt.text(x_mid, y_mid, texte_poids, fontsize=8, color='black', 
                     ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Affichage de la fenêtre
    plt.show()