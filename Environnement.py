import pygame
import random
import math
import numpy as np
from Agent import Agent
from GraphVisualisation import show_sbn_graph

# ==========================================================
# CONFIGURATION DE LA SIMULATION
# ==========================================================

# Paramètre de la simulation
PARAMS = {
    "TEST_NAME": "Extinction_Initiale",
    "SEED": 42,
    "NUM_AGENTS": 1000,
    "BASE_ENERGY": 100,
    "DIVISION_ENERGY": 600,
    "MODE_FOOD": 1,
    "N_BOOST": 10,
    "QUANTITE_BOOST": 1,
    "PROBA_DELETION": 0.01,
    "PROBA_INSERTION": 0.02,
    "VALEUR_MAX_POIDS": 3
}

# On récupère les paramètre du dictionnaire si ils existent sinon on met les valeurs par défauts
# --- PARAMÈTRES GÉNÉRAUX ---
TEST_NAME = PARAMS.get("TEST_NAME", "Extinction_Initiale") # Nom de la session ou de l'expérience en cours
SEED = PARAMS.get("SEED", 42)                            # Graine de génération aléatoire (assure que la simulation est reproductible)

# --- MOTEUR ET AFFICHAGE ---
WIDTH = PARAMS.get("WIDTH", 1280)                        # Largeur de la fenêtre de simulation en pixels
HEIGHT = PARAMS.get("HEIGHT", 720)                       # Hauteur de la fenêtre de simulation en pixels
N_ITER = PARAMS.get("N_ITER", 1)                         # Nombre de mises à jour logiques par image (accélère le temps simulé sans affecter l'affichage)
FPS = PARAMS.get("FPS", 60)                              # Limite d'images par seconde pour le rendu visuel
DISPLAY_FPS = PARAMS.get("DISPLAY_FPS", 0)               # Active (1) ou désactive (0) l'affichage du compteur visuel des FPS

# --- PHYSIQUE ET CAPACITÉS DES AGENTS ---
TAILLE_AGENT = PARAMS.get("TAILLE_AGENT", 5)                        # Rayon d'affichage du cercle représentant l'agent
ROTATE_DEG = PARAMS.get("ROTATE_DEG", 10)                           # Vitesse de rotation : nombre de degrés modifiés quand l'agent décide de tourner
VISION_ANGLE = PARAMS.get("VISION_ANGLE", 45)                       # Ouverture du cône de vision de l'agent (en degrés)
DISTANCE_MANGER = PARAMS.get("DISTANCE_MANGER", TAILLE_AGENT*2)     # Portée maximale à laquelle l'agent peut attaquer/manger
DISTANCE_VISION = PARAMS.get("DISTANCE_VISION", DISTANCE_MANGER)    # Portée maximale à laquelle l'agent détecte les autres entités

# --- DYNAMIQUE DE POPULATION ET ÉNERGIE ---
NUM_AGENTS = PARAMS.get("NUM_AGENTS", 1000)              # Taille de la population de départ générée aléatoirement
BASE_ENERGY = PARAMS.get("BASE_ENERGY", 100)             # Capital énergétique de départ pour les premiers agents
DIVISION_ENERGY = PARAMS.get("DIVISION_ENERGY", 600)     # Palier d'énergie requis pour déclencher la reproduction par division
MODE_FOOD = PARAMS.get("MODE_FOOD", 1)                   # Comportement d'alimentation (1 : Photosynthèse, 2 : Nourriture au sol)
N_BOOST = PARAMS.get("N_BOOST", 10)                      # Fréquence (en itérations) d'apport passif d'énergie (photosynthèse)
QUANTITE_BOOST = PARAMS.get("QUANTITE_BOOST", 1)         # Montant de l'énergie récupérée lors du boost passif

# --- MUTATIONS DU CERVEAU (RÉSEAU DE NEURONES) ---
PROBA_DELETION = PARAMS.get("PROBA_DELETION", 0.01)      # Probabilité qu'un agent perde un nœud neuronal lors d'une mutation
PROBA_INSERTION = PARAMS.get("PROBA_INSERTION", 0.02)    # Probabilité qu'un agent développe un nouveau nœud neuronal
VALEUR_MAX_POIDS = PARAMS.get("VALEUR_MAX_POIDS", 3)     # Valeur absolue maximale des connexions créées entre les neurones
    
# Paramètre de la grille
CELL_SIZE = DISTANCE_VISION * 1.2  # Doit être >= DISTANCE_VISION (on met distance_vision + 20% pour avoir de la marge d'erreur)

random.seed(SEED) # On applique la seed a random (cela marchera peut importe ou random est appelé (n'importe quel classe))
np.random.seed(SEED) # Pareil pour le random de numpy

# Dashboard d'information
DASHBOARD_SIZE = 50

def draw_dashboard(screen, clock, agents, total_steps, params):
    # 1. Dessin du fond du bandeau
    pygame.draw.rect(screen, (20, 20, 20), (0, 0, WIDTH, DASHBOARD_SIZE-5)) # -5 pour laisser un petit gap
    pygame.draw.line(screen, (150, 150, 150), (0, DASHBOARD_SIZE-5), (WIDTH, DASHBOARD_SIZE-5), 2) # -5 pour laisser un petit gap

    # 2. Informations à afficher
    fps = int(clock.get_fps())
    pop = len(agents)
    mode_txt = "Photosynthèse" if params["MODE_FOOD"] == 1 else "Chasse"
    
    # Rendu des textes
    txt_test = font.render(f"TEST: {params['TEST_NAME']}", True, (255, 255, 255))
    txt_pop  = font.render(f"POPULATION: {pop}", True, (0, 255, 100) if pop > 0 else (255, 50, 50))
    txt_step = font.render(f"STEPS: {total_steps}", True, (200, 200, 200))
    txt_mode = font.render(f"MODE: {mode_txt}", True, (100, 200, 255))
    txt_fps  = font.render(f"FPS: {fps}", True, (255, 255, 0))

    # 3. Positionnement sur le bandeau
    screen.blit(txt_test, (20, 15))
    screen.blit(txt_mode, (300, 15))
    screen.blit(txt_step, (550, 15))
    screen.blit(txt_pop,  (750, 15))
    screen.blit(txt_fps,  (WIDTH - 100, 15))

# La grille est un dictionnaire pour plus de flexibilité
grid = {}

# On applique ces méthode car boucler sur tout les agents alors que certain sont très loin nous consommaient trop de ressources (bcp trop !)
def update_grid(agents):
    grid = {}
    for agent in agents:
        # Calcul de l'index de la case
        cx = int(agent.x // CELL_SIZE)
        cy = int(agent.y // CELL_SIZE)
        
        cell_key = (cx, cy)
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(agent)
    return grid

def get_neighbors(agent, grid):
    neighbors = []
    cx = int(agent.x // CELL_SIZE)
    cy = int(agent.y // CELL_SIZE)

    # On boucle sur les 9 cases (celle de l'agent + les 8 voisines)
    for i in range(cx - 1, cx + 2):
        for j in range(cy - 1, cy + 2):
            if (i, j) in grid:
                neighbors.extend(grid[(i, j)])
    
    return neighbors

# --- INITIALISATION PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + DASHBOARD_SIZE))
pygame.display.set_caption("Simulation SBN Evolution")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 18)

# Fonction permettant l'affichage des FPS (images par seconde)
def display_fps(screen, clock):
    # Récupère les FPS réels calculés par Pygame
    fps = str(int(clock.get_fps()))
    fps_text = font.render(f"FPS: {fps}", True, (255, 255, 0)) # Jaune
    screen.blit(fps_text, (10, 10)) # Affichage en haut à gauche

# Liste stockant les agents
agents = [Agent(i, random.randint(0, WIDTH), random.randint(0, HEIGHT), BASE_ENERGY, ROTATE_DEG, WIDTH, HEIGHT) for i in range(NUM_AGENTS)]

running = True
show_graphics = True # Variable permettant d'afficher ou non la simulation (pour aller plus vite si nécessaire)
total_steps = 0 # Variable stockant le nombre de pas de la simulation

new_id = NUM_AGENTS # Nouveau id a incrémenter à partir du nombre d'agent initiaux (pour la descendance)

is_paused = False # Variable permettant de mettre en pause la simulation

# On pré-calcule le carré de la distance pour éviter les racines carrées (LENT)
DIST_MANGER_SQ = DISTANCE_MANGER * DISTANCE_MANGER
DISTANCE_VISION_SQ = DISTANCE_VISION**2

while running:
    # 1. Paramétrage des touches de la simulation
    for event in pygame.event.get():    
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE: # Espace pour faire Pause/Dépause
                is_paused = not is_paused
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 1 correspond au clic gauche
                is_paused = True
                mx, my = pygame.mouse.get_pos()
                # On ramène le clic de la souris dans le référentiel de la simulation
                my = my - DASHBOARD_SIZE
                
                rayon_sq = TAILLE_AGENT * TAILLE_AGENT
                # On cherche quel agent a été cliqué
                for agent in agents:
                    if not agent.alive: continue
                    dx = mx - agent.x
                    dy = my - agent.y
                    
                    # Test de collision Cercle / Point sans racine carrée
                    if (dx*dx + dy*dy) <= rayon_sq:
                        print(f"Ouverture du cerveau de l'agent {agent.id}")
                        # Appel de la fonction de visualisation du réseau sbn
                        show_sbn_graph(agent.id, agent.sbn)
                        # On a trouvé l'agent cliqué, inutile de tester les autres
                        break
    
    # 2. LOGIQUE
    if not is_paused:
        for _ in range(N_ITER):
            total_steps += 1
            new_enfants = []
            
            # Mélanger pour l'équité
            random.shuffle(agents)
            
            spatial_grid = update_grid(agents)
            
            for agent in agents:
                if not agent.alive:
                    continue
                
                # Boost d'énergie tout les N_BOOST pas de temps
                if MODE_FOOD == 1:
                    if total_steps % N_BOOST == 0:
                        agent.energy += QUANTITE_BOOST
                
                # On récupère la liste des agents proche du notre
                neighbors = get_neighbors(agent, spatial_grid)
                # On enregistre dans l'agent agent.proie_potentielle si une cible est a sa portée
                agent.sense(neighbors, DISTANCE_VISION_SQ, DIST_MANGER_SQ, VISION_ANGLE)
                
                # Décision manger
                action_eat = agent.update(agent.vision_input, PROBA_DELETION, PROBA_INSERTION, VALEUR_MAX_POIDS)
                
                # Manger
                # On utilise la proie qu'on a trouvée dans la boucle de vision
                if action_eat and (agent.proie_potentielle is not None):
                    victim = agent.proie_potentielle
                    # Vérification de sécurité (la victime est peut-être morte entre temps dans ce tour)
                    if victim.alive:
                        victim.alive = False
                        agent.energy += victim.energy

                # Division
                if agent.energy >= DIVISION_ENERGY:
                    # On définit la position de l'enfant lors de la division bornée entre les limites de la simulation
                    if(random.random()< 0.5): 
                        new_pos_x = min(agent.x + 15, WIDTH)
                    else:
                        new_pos_x = max(agent.x - 15, 0)
                    if (random.random() <0.5):
                        new_pos_y = min(agent.y + 15, HEIGHT)
                    else:
                        new_pos_y = max(agent.y - 15, 0)
                    enfant = agent.division(new_id, new_pos_x, new_pos_y)
                    new_id += 1
                    new_enfants.append(enfant)
            
            # Nettoyage rapide
            agents = [a for a in agents if a.alive]
            agents.extend(new_enfants)
        
    # 3. AFFICHAGE (RENDU)
    if show_graphics:
        screen.fill((0, 0, 0))
        # On crée un calque transparent (SRCALPHA permet de gérer l'opacité)
        #overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        # On le vide à chaque frame
        #overlay.fill((0, 0, 0, 0))
        
        # Affichage des agents
        for agent in agents:
            agent.draw(screen, screen, TAILLE_AGENT, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY, DASHBOARD_SIZE)
        
        #screen.blit(overlay, (0, 0))
        # On affiche les fps
        if(DISPLAY_FPS):  display_fps(screen, clock)
        # On affiche le dashboard
        draw_dashboard(screen, clock, agents, total_steps, PARAMS)
        # On rafraîchit l'écran une fois après la boucle des agents
        pygame.display.flip()
    
    clock.tick(FPS)

pygame.quit()