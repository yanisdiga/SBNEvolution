import pygame
import random
import math
from Agent import Agent

WIDTH, HEIGHT = 1280, 720
N_ITER = 1
NUM_AGENTS = 200
BASE_ENERGY = 100
ROTATE_DEG = 10
DIVISION_ENERGY = 400
TAILLE_AGENT = 5
DISTANCE_MANGER = TAILLE_AGENT*2 # Distance a laquel un agent peut manger un autre
PROBA_DELETION = 0.01
PROBA_INSERTION = 0.02
VALEUR_MAX_POIDS = 3
N_BOOST = 1 # Nombre d'itération avant d'avoir un boost
QUANTITE_BOOST = 1.5 # Quantité du boost d'énergie
VISION_ANGLE = 45 # en degrées
DISTANCE_VISION = DISTANCE_MANGER # distance a laquel un agent voit (pour l'instant même distance que pour manger)

CELL_SIZE = DISTANCE_VISION * 1.2  # Doit être >= DISTANCE_VISION (on met distance_vision + 20% pour avoir de la marge d'erreur)
cols = WIDTH // CELL_SIZE + 1
rows = HEIGHT // CELL_SIZE + 1

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
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 18)
def display_fps(screen, clock):
    # Récupère les FPS réels calculés par Pygame
    fps = str(int(clock.get_fps()))
    fps_text = font.render(f"FPS: {fps}", True, (255, 255, 0)) # Jaune
    screen.blit(fps_text, (10, 10)) # Affichage en haut à gauche

agents = [Agent(i, random.randint(0, WIDTH), random.randint(0, HEIGHT), BASE_ENERGY, ROTATE_DEG, WIDTH, HEIGHT) for i in range(NUM_AGENTS)]

running = True
show_graphics = True
total_steps = 0

new_id = NUM_AGENTS

is_paused = False

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
                    if(random.random()< 0.5): 
                        new_pos_x = agent.x + 15
                    else:
                        new_pos_x = agent.x - 15
                    if (random.random() <0.5):
                        new_pos_y = agent.y + 15
                    else:
                        new_pos_y = agent.y - 15
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
            agent.draw(screen, screen, TAILLE_AGENT, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY)
        
        #screen.blit(overlay, (0, 0))
        # On affiche les fps
        display_fps(screen, clock)
        # On rafraîchit l'écran une fois après la boucle des agents
        pygame.display.flip()
    
    clock.tick(60)

pygame.quit()