import pygame
import random
import math
from Agent import Agent

WIDTH, HEIGHT = 1280, 720
N_ITER = 1
NUM_AGENTS = 100
BASE_ENERGY = 200
ROTATE_DEG = 10
DIVISION_ENERGY = 600
TAILLE_AGENT = 5
DISTANCE_MANGER = TAILLE_AGENT*2 # Distance a laquel un agent peut manger un autre
PROBA_DELETION = 0.01
PROBA_INSERTION = 0.02
VALEUR_MAX_POIDS = 3
N_BOOST = 1 # Nombre d'itération avant d'avoir un boost
QUANTITE_BOOST = 1 # Quantité du boost d'énergie
VISION_ANGLE = 45 # en degrées
DISTANCE_VISION = DISTANCE_MANGER # distance a laquel un agent voit (pour l'instant même distance que pour manger)

### A REINITIALISER N_BOOST, BASE_ENERGY, NUM_AGENT

# --- INITIALISATION PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

agents = [Agent(i, random.randint(0, WIDTH), random.randint(0, HEIGHT), BASE_ENERGY, ROTATE_DEG, WIDTH, HEIGHT) for i in range(NUM_AGENTS)]

running = True
show_graphics = True
total_steps = 0

new_id = 1000

while running:
    # 1. INDISPENSABLE : Dire à l'OS que le programme répond
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # 2. LOGIQUE
    for _ in range(N_ITER):
        total_steps += QUANTITE_BOOST
        new_enfants = []
        
        # Mélanger pour l'équité
        random.shuffle(agents)

        # On pré-calcule le carré de la distance pour éviter les racines carrées (LENT)
        DIST_MANGER_SQ = DISTANCE_MANGER * DISTANCE_MANGER
        DISTANCE_VISION_SQ = DISTANCE_VISION**2
        for agent in agents:
            if not agent.alive:
                continue
            
            # Boost d'énergie tout les N_BOOST pas de temps
            if total_steps % 10 == 0:
                agent.energy += 1
            
            # On regarde si l'agent voit une cible
            vision_input = 0
            proie_potentielle = None
            
            # On cherche les voisins
            for other in agents:
                if other is not agent and other.alive:
                    # Calcul de distance au carré
                    distance_x = other.x - agent.x
                    distance_y = other.y - agent.y
                    dist_sq = distance_x*distance_x + distance_y*distance_y
                    
                    if dist_sq < DISTANCE_VISION_SQ:
                        # On vérifie si l'agent regarde bien vers la cible
                        
                        # On calcule l'angle vers la cible (en radians) (arc tangente)
                        angle_vers_cible = math.atan2(distance_y, distance_x)
                        
                        # On récupère l'angle actuel de l'agent (converti en radians)
                        angle_regard = math.radians(agent.angle)
                        
                        # On calcul la différence
                        diff = angle_vers_cible - angle_regard
                        
                        # Normalisation
                        # On ramène automatiquement la différence entre -PI et +PI
                        # Ex: -350° devient +10°
                        diff = (diff + math.pi) % (2 * math.pi) - math.pi
    
                        # Si la différence est petite alors c'est "devant"
                        if abs(diff) < math.radians(VISION_ANGLE):
                            vision_input = 1
                            # On vérifie que la cible est assez près pour la manger
                            if dist_sq < DIST_MANGER_SQ:
                                proie_potentielle = other
                            break
            
            # Décision manger
            action_eat = agent.update(vision_input, PROBA_DELETION, PROBA_INSERTION, VALEUR_MAX_POIDS)
            
            # Manger
            # On utilise la proie qu'on a trouvée dans la boucle de vision
            if action_eat and (proie_potentielle is not None):
                victim = proie_potentielle
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
        
        for agent in agents:
            if agent.alive:
                pos = (int(agent.x), int(agent.y))
                
                # Couleur : du Rouge (mort) au Vert (vie)
                ratio = max(0, min(agent.energy / BASE_ENERGY, 1))
                red = int(255 * (1 - ratio))
                green = int(255 * ratio)
                color = (red, green, 0)
                pygame.draw.circle(screen, color, pos, TAILLE_AGENT)
                
                # Pour voir la direction ou les agents regarde
                # end_x = agent.x + 10 * math.cos(math.radians(agent.angle))
                # end_y = agent.y + 10 * math.sin(math.radians(agent.angle))
                # pygame.draw.line(screen, (255, 255, 255), pos, (end_x, end_y), 1)
                
                portee = DISTANCE_VISION
                rad_gauche = math.radians(agent.angle - VISION_ANGLE)
                rad_droite = math.radians(agent.angle + VISION_ANGLE)
                
                p_left = (agent.x + portee * math.cos(rad_gauche), agent.y + portee * math.sin(rad_gauche))
                p_right = (agent.x + portee * math.cos(rad_droite), agent.y + portee * math.sin(rad_droite))
                
                # Couleur adaptative
                c_cone = (255, 255, 255, 20) # Blanc très transparent par défaut
                if agent.sbn.states[0] == 1: c_cone = (255, 0, 0, 40) # Rouge si "Je vois"
                
                # On dessine sur le calque (pas sur l'écran direct)
                pygame.draw.polygon(screen, c_cone, [pos, p_left, p_right])
                
                # Voir si l'agent mange
                if agent.sbn.states[1] == 1:
                    pygame.draw.circle(screen, (255,255,255), pos, TAILLE_AGENT+3, 1)
                    
        # On rafraîchit l'écran une fois après la boucle des agents
        pygame.display.flip()
    
    clock.tick(60)

pygame.quit()