import pygame
import random
import math
import numpy as np
from Agent import Agent
from GraphVisualisation import *
from Interface import *
from SpatialGrid import *
from Food import Food

# ==========================================================
# CONFIGURATION DE LA SIMULATION
# ==========================================================

# Paramètre de la simulation
PARAMS = {
    "TEST_NAME": "Infinite_Life_Cycle",
    "SEED": 42,
    "NUM_AGENTS": 200,
    "BASE_ENERGY": 1000,
    "DIVISION_ENERGY": 2000,
    "PROBA_DELETION": 0.05,
    "PROBA_INSERTION": 0.1,
    "PROBA_EVOLUTION": 0.05,
    "VALEUR_MAX_POIDS": 3,
    "DISTANCE_VISION": 70,
    "VISION_ANGLE": 30,
    "MODE_FOOD": 3,
    "COST_NEURON": 0.5,
    "COST_MOVE": 1,
    "COST_ROTATE": 1,  
    "COST_EAT": 0,   
    "COST_METABOLISM": 0,   
    "NUM_FOOD": 20,
    "ALIMENTATION_BOOST": 1,
    "PHOTOSYNTHESE_BOOST": 5,
    "PHOTOSYNTHESE_MIN": 0,
    "PHOTOSYNTHESE_DECREASE": 0.0004,
    "PHOTOSYNTHESE_INTERVAL": 1,
    "PHOTOSYNTHESE_INTERVAL_UPDATE": 1,
    "DIGESTION_INTERVAL": 30,
    "DIGESTION_RATE": 3,
    "ACTIVE_CORPSE": False
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
PHOTOSYNTHESE_INTERVAL = PARAMS.get("PHOTOSYNTHESE_INTERVAL", 10)   # Fréquence (en itérations) d'apport passif d'énergie (photosynthèse)
ALIMENTATION_BOOST = PARAMS.get("ALIMENTATION_BOOST", 1)          # Montant de l'énergie récupérée lorsqu'un agent mange au sol
PHOTOSYNTHESE_BOOST = PARAMS.get("PHOTOSYNTHESE_BOOST", 1)  # Montant de l'énergie récupérée lors du boost passif
PHOTOSYNTHESE_DECREASE = PARAMS.get("PHOTOSYNTHESE_DECREASE", 0.01)
PHOTOSYNTHESE_INTERVAL_UPDATE = PARAMS.get("PHOTOSYNTHESE_INTERVAL_UPDATE", 100)
PHOTOSYNTHESE_MIN = PARAMS.get("PHOTOSYNTHESE_MIN", 0)
NUM_FOOD = PARAMS.get("NUM_FOOD", 100)
COST_MOVE = PARAMS.get("COST_MOVE", 1)
COST_ROTATE = PARAMS.get("COST_ROTATE", 1)
COST_EAT = PARAMS.get("COST_EAT", 1)
COST_NEURON = PARAMS.get("COST_NEURON", 1)
COST_METABOLISM = PARAMS.get("COST_METABOLISM", 0.05)
DIGESTION_RATE = PARAMS.get("DIGESTION_RATE", 5)
DIGESTION_INTERVAL = PARAMS.get("DIGESTION_INTERVAL", 10)
ACTIVE_CORPSE = PARAMS.get("ACTIVE_CORPSE", True)

# --- MUTATIONS DU CERVEAU (RÉSEAU DE NEURONES) ---
PROBA_DELETION = PARAMS.get("PROBA_DELETION", 0.01)      # Probabilité qu'un agent perde un nœud neuronal lors d'une mutation
PROBA_INSERTION = PARAMS.get("PROBA_INSERTION", 0.02)    # Probabilité qu'un agent développe un nouveau nœud neuronal
PROBA_EVOLUTION = PARAMS.get("PROBA_EVOLUTION", 0.02)    # Probabilité qu'un agent développe un nouveau nœud neuronal
VALEUR_MAX_POIDS = PARAMS.get("VALEUR_MAX_POIDS", 3)     # Valeur absolue maximale des connexions créées entre les neurones

# --- STATISTIQUES ET VISUALISATION ---
TRACKING_ID = PARAMS.get("TRACKING_ID", None)
    
# Paramètre de la grille
CELL_SIZE = DISTANCE_VISION * 1.2  # Doit être >= DISTANCE_VISION (on met distance_vision + 20% pour avoir de la marge d'erreur)

random.seed(SEED) # On applique la seed a random (cela marchera peut importe ou random est appelé (n'importe quel classe))
np.random.seed(SEED) # Pareil pour le random de numpy

# Dashboard d'information
DASHBOARD_SIZE = 50

# --- INITIALISATION PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + DASHBOARD_SIZE))
overlay = pygame.Surface((WIDTH, HEIGHT + DASHBOARD_SIZE), pygame.SRCALPHA)
pygame.display.set_caption("Simulation SBN Evolution")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 18)

# Liste stockant les agents
agents = [Agent(i, random.randint(0, WIDTH), random.randint(0, HEIGHT), BASE_ENERGY, ROTATE_DEG, WIDTH, HEIGHT, COST_ROTATE, COST_MOVE, COST_EAT, COST_NEURON, COST_METABOLISM, DIGESTION_RATE, DIGESTION_INTERVAL) for i in range(NUM_AGENTS)]
foods = [Food(random.randint(0, WIDTH), random.randint(0, HEIGHT), energy=ALIMENTATION_BOOST) for _ in range(NUM_FOOD)]

running = True
show_graphics = True # Variable permettant d'afficher ou non la simulation (pour aller plus vite si nécessaire)
total_steps = 0 # Variable stockant le nombre de pas de la simulation

new_id = NUM_AGENTS # Nouveau id a incrémenter à partir du nombre d'agent initiaux (pour la descendance)

is_paused = True # Variable permettant de mettre en pause la simulation
temps_simule_ms = 0

vision_cone = True

# On pré-calcule le carré de la distance pour éviter les racines carrées (LENT)
DIST_MANGER_SQ = DISTANCE_MANGER * DISTANCE_MANGER
DISTANCE_VISION_SQ = DISTANCE_VISION**2

# On récupère la valeur original de la photosynthèse
PHOTOSYNTHESE_INITIAL_BOOST = PHOTOSYNTHESE_BOOST

# Listes pour stocker l'historique
stats_steps = []
stats_pop = []
stats_size = []
stats_energy = []
stats_node_activated = []

hiver = False

while running:
    # 1. Paramétrage des touches de la simulation
    for event in pygame.event.get():    
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE: # Espace pour faire Pause/Dépause
                is_paused = not is_paused
                if not show_graphics: show_graphics_off(screen, font, WIDTH, HEIGHT, is_paused, DASHBOARD_SIZE)
            elif event.key == pygame.K_g:   # Touche G pour activer/désactiver le rendu
                show_graphics = not show_graphics     
                if not show_graphics: show_graphics_off(screen, font, WIDTH, HEIGHT, is_paused, DASHBOARD_SIZE)
            elif event.key == pygame.K_v:   # Touche G pour activer/désactiver le rendu graphique des cones de vision
                vision_cone = not vision_cone
                    
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
                        # Appel de la fonction de visualisation du réseau sbn
                        show_sbn_graph(agent.id, agent.sbn)
                        # On a trouvé l'agent cliqué, inutile de tester les autres
                        break
    
    # 2. LOGIQUE
    if not is_paused:
        for _ in range(N_ITER):
            total_steps += 1
            temps_simule_ms += clock.get_time()
            new_enfants = []
            
            # Mélanger pour l'équité
            random.shuffle(agents)
            
            spatial_grid = update_grid(agents, foods, CELL_SIZE, MODE_FOOD)
            
            eat_intentions = {} # Dictionnaire stockant l'intention de manger de chaque agent

            if MODE_FOOD == 3:
                # if total_steps > 10000 and total_steps % 100 == 0:
                if total_steps % PHOTOSYNTHESE_INTERVAL_UPDATE == 0:
                    if PHOTOSYNTHESE_BOOST <= PHOTOSYNTHESE_MIN:
                        hiver = False
                    elif PHOTOSYNTHESE_BOOST >= PHOTOSYNTHESE_INITIAL_BOOST: 
                        hiver = True
                    if hiver:
                        PHOTOSYNTHESE_BOOST = max(PHOTOSYNTHESE_BOOST-PHOTOSYNTHESE_DECREASE, PHOTOSYNTHESE_MIN)
                    else:
                        PHOTOSYNTHESE_BOOST = min(PHOTOSYNTHESE_BOOST+PHOTOSYNTHESE_DECREASE, PHOTOSYNTHESE_INITIAL_BOOST)
            
            for agent in agents:
                if not agent.alive:
                    continue
                
                # Boost d'énergie tout les N_BOOST pas de temps
                if MODE_FOOD == 1 or MODE_FOOD == 3:
                    if total_steps % PHOTOSYNTHESE_INTERVAL == 0:
                        agent.energy += PHOTOSYNTHESE_BOOST
                
                # On récupère la liste des cases proche de nous
                neighbors = get_neighbors(agent, spatial_grid, CELL_SIZE)
                # On enregistre dans l'agent agent.proie_potentielle si une cible est a sa portée
                agent.sense(neighbors, DISTANCE_VISION_SQ, DIST_MANGER_SQ, VISION_ANGLE)
                
                # Décision manger
                action_eat = agent.update(agent.vision_input, PROBA_DELETION, PROBA_INSERTION, PROBA_EVOLUTION, VALEUR_MAX_POIDS)
                
                # Manger
                # On utilise la proie qu'on a trouvée dans la boucle de vision
                if action_eat and (agent.proie_potentielle is not None):
                    victim = agent.proie_potentielle
                    # Vérification de sécurité
                    if victim.alive:
                        if isinstance(victim, Agent):
                            eat_intentions[agent] = agent.proie_potentielle
                        elif isinstance(victim, Food):
                            agent.eat(victim)
                            foods.remove(victim) # On enleve la nourriture du sol
                            #foods.append(Food(random.randint(0, WIDTH), random.randint(0, HEIGHT))) # On la fais réapparaître ailleurs
                            
                # Digestion
                if agent.stomach > 0 and agent.step % agent.digestion_interval==0:
                    waste = agent.digestion()
                    new_food = Food(agent.x, agent.y, energy=waste)
                    foods.append(new_food)
                    
                # Division
                if agent.energy >= DIVISION_ENERGY:
                    enfant = agent.division(new_id, VISION_ANGLE, PROBA_DELETION, PROBA_INSERTION, VALEUR_MAX_POIDS)
                    new_id += 1
                    new_enfants.append(enfant)
            
            # Vérification de conflit lors de l'action "manger"
            for agent, victim in eat_intentions.items():
                # On vérifie que l'agent et la victime sont bien toujours vivant
                if not agent.alive or not victim.alive: continue
                
                # CAS 1 : Duel
                if eat_intentions.get(victim) == agent:
                    if agent.energy > victim.energy:
                        agent.eat(victim)
                    elif agent.energy == victim.energy:
                        if random.random()>0.5: 
                            agent.eat(victim)
                        else: 
                            victim.eat(agent)
                    else:
                        victim.eat(agent)
                # CAS 2 : Normal
                else:
                    agent.eat(victim)
            
            # Nettoyage rapide
            if ACTIVE_CORPSE:
                for a in agents:
                    # On cible ceux qui viennent de mourir (famine ou vieillesse) 
                    # et qui n'ont pas été "vidés" par un prédateur
                    if not a.alive and a.stomach > 0:
                        # On dépose le contenu de leur estomac au sol
                        foods.append(Food(a.x, a.y, energy=a.stomach))
                        # On vide l'estomac pour éviter les doublons si le code repasse dessus
                        a.stomach = 0 

            # Ensuite, on fait ton nettoyage habituel
            agents = [a for a in agents if a.alive]
            agents.extend(new_enfants)
        
    # 3. AFFICHAGE (RENDU)
    if show_graphics:
        screen.fill((0, 0, 0))
        # On le vide à chaque frame
        overlay.fill((0, 0, 0, 0))
        
        # Affichage des agents
        for agent in agents:
            is_tracked = (agent.id == TRACKING_ID)
            if DISTANCE_VISION>20: # On applique l'opacité sur le cone de vision pour plus de visibilité (si le cone est assez grand sinon on le met en blanc car il ne sera pas visible)
                agent.draw(screen, overlay, TAILLE_AGENT, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY, DASHBOARD_SIZE, vision_cone=vision_cone, tracking=is_tracked) # On applique l'opacité sur le cone de vision pour plus de visibilité
            else:
                agent.draw(screen, screen, TAILLE_AGENT, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY, DASHBOARD_SIZE, vision_cone=vision_cone, tracking=is_tracked)
            
        if MODE_FOOD == 2 or MODE_FOOD == 3:
            for food in foods:
                food.draw(screen, DASHBOARD_SIZE)
        
        # On fusionne le calque avec le screen
        screen.blit(overlay, (0, 0))
        # On affiche le dashboard
        draw_dashboard(screen, clock, agents, total_steps, PARAMS, WIDTH, DASHBOARD_SIZE, font, temps_simule_ms)
        
        if is_paused: graphics_pause(screen, font, DASHBOARD_SIZE)
        
        # On rafraîchit l'écran une fois après la boucle des agents
        pygame.display.flip()
        clock.tick(FPS)
        
    # 4. Statistiques
    if total_steps % 100 == 0 and len(agents) > 0:
        stats_steps.append(total_steps)
        stats_pop.append(len(agents))
        agent_max = max(a.step for a in agents)
        age_max = max(a.step for a in agents)
        
        # Calcul des moyennes
        moyenne_noeuds = sum(a.sbn.num_nodes for a in agents) / len(agents)
        moyenne_energie = sum(a.energy for a in agents) / len(agents)
        moyenne_noeuds_actif = sum(sum(a.sbn.states) for a in agents) / len(agents)
        stats_size.append(moyenne_noeuds)
        stats_energy.append(moyenne_energie)
        stats_node_activated.append(moyenne_noeuds_actif)
        
        if not show_graphics:
            secondes_ecoulees = temps_simule_ms // 1000
            fps_actuel = int(clock.get_fps())
            print(f"[{secondes_ecoulees}s] Step: {total_steps} | Pop: {len(agents)} | Énergie Moy: {moyenne_energie:.0f} | Noeuds Moy: {moyenne_noeuds:.1f} | Âge Max: {age_max}")

pygame.quit()

show_simulation_summary(stats_steps, stats_pop, stats_size, stats_energy, stats_node_activated)