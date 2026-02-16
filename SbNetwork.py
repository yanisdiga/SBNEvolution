import numpy as np
import random

class SbNetwork:
    def __init__(self):
        self.num_nodes=8
        # Création de la matrice des états
        self.states = np.zeros(self.num_nodes, dtype=int)
        # Signification des positions dans le tableau
        # 0. Oeil
        # 1. Bouche
        # 2. Nageoire F (Forward)
        # 3. Nageoire R (Rotate)
        # 4. R1
        # 5. R2
        # 6. R3
        # 7. R4
        
        # Création de la matrice des poids
        self.weights = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        
        # Initialisation des états
        self.states[2] = 1 # Nageoire F est tout le temps activé
        self.states[3] = 1 # Nageoire R est activé à l'initialisation
        
        # Initialisation des poids
        self.weights[0, 1] = 1 # Oeil active la bouche
        self.weights[2, 2] = 1 # Negeoire F active Nageoire F
        self.weights[3, 3] = -1 # R s'auto inhibe
        self.weights[3, 4] = 1 # R active R1 ...
        self.weights[4, 4] = -1
        self.weights[4, 5] = 1
        self.weights[5, 5] = -1
        self.weights[5, 6] = 1
        self.weights[6, 6] = -1
        self.weights[6, 7] = 1
        self.weights[7, 7] = -1
        self.weights[7, 3] = 1
        
    def step(self, oeil):
        # On récupère bien la valeur de l'oeil par rapport a l'environnement
        self.states[0] = oeil
        
        # Calcul de la somme pondérée pour tous les noeuds d'un coup (Produit matriciel)
        x = self.states @ self.weights
        
        # On applique un seuil (si x > 0, on met 1, sinon 0)
        new_states = (x > 0).astype(int)
        
        # On remet la bonne valeur de l'oeil
        new_states[0] = oeil
        
        # Mise à jour synchrone
        self.states = new_states
        
        # On return que les états actionneurs
        return self.states[1], self.states[2], self.states[3]

    def mutation(self, pd, pi, wmax):
        if (random.random() < pd):
            self.deletion()
        if (random.random() < pi):
            self.insertion(wmax)
            
    def insertion(self, wmax):
        n = self.num_nodes
        # On ajoute un noeud au réseau de neuronnes
        self.states = np.append(self.states, random.randint(0, 1))
        # On crée une nouvelle matrice de poids d'un cran plus grande
        new_weights = np.zeros((n+1, n+1), dtype=int)
        # On copie l'ancienne matrice dans la nouvelle
        new_weights[:n, :n] = self.weights
        # On rempli les poids du nouvelle état entre [-wmax, wmax]
        # On commence par la dernière colonne
        new_weights[:, n] = np.random.randint(-wmax, wmax+1, size=n+1) # +1 car [start, stop[
        # La dernière ligne
        new_weights[n, :] = np.random.randint(-wmax, wmax+1, size=n+1) # +1 car [start, stop[
        
        # On remplace l'ancienne matrice de poids par la nouvelle
        self.weights = new_weights
        self.num_nodes += 1 # On met a jour le nombre de noeud
        
    def deletion(self):
        if self.num_nodes <=4:
            return
        
        node_to_delete = random.randint(4, self.num_nodes-1)
        self.states = np.delete(self.states, node_to_delete)
        self.weights = np.delete(self.weights, node_to_delete, axis=0)
        self.weights = np.delete(self.weights, node_to_delete, axis=1)
        self.num_nodes -=1
        
        