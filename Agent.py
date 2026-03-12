import random
import math
import pygame
import numpy as np
import random
from SbNetwork import SbNetwork

class Agent:
    def __init__(self, id, x, y, energy, rotate_deg, env_width, env_height, cost_rotate, cost_move, cost_eat, cost_neuron, cost_metabolism, digestion_rate, digestion_interval):
        self.id = id
        self.x = x
        self.y = y
        self.energy = energy
        self.rotate_deg = rotate_deg
        self.env_width = env_width
        self.env_height = env_height
        self.sbn = SbNetwork()
        self.speed = 2
        self.angle = random.uniform(0, 360)
        
        # Variable stockant une information liée a la simulation
        self.alive = True
        self.proie_potentielle = None
        self.vision_input = 0
        self.step = 0
        
        # Cout des actions
        self.cost_rotate = cost_rotate
        self.cost_move = cost_move
        self.cost_eat = cost_eat
        self.cost_neuron = cost_neuron
        self.cost_metabolism = cost_metabolism
        
        # Digestion
        self.stomach = 0
        self.digestion_rate = digestion_rate
        self.digestion_quantity = 0
        self.digestion_interval = digestion_interval
        
    def move(self):
        # On convertis l'angle en radian pour les calculs
        rad = math.radians(self.angle)
    
        dx = self.speed * math.cos(rad)
        dy = self.speed * math.sin(rad)
        
        # On crée des nouvelles position
        new_x = self.x + dx
        new_y = self.y + dy
        
        # On met a jour les nouvelles position en prenant en compte les limites de l'environnement
        self.x = max(0, min(new_x, self.env_width))
        self.y = max(0, min(new_y, self.env_height))
        
    def rotate(self):
        self.angle += self.rotate_deg
        self.angle %= 360 # On garde l'angle entre 0 et 360
        
    def eat(self, victim):
        victim.alive = False
        new_energy = victim.energy
        self.energy += new_energy*0.6
        self.stomach += new_energy*0.4 
        self.digestion_quantity = self.stomach/self.digestion_rate   
    
    def digestion(self):
        waste = min(self.digestion_quantity, self.stomach)
        self.stomach -= waste
        return waste
    
    def division(self, id, vision_angle, pd, pi, wmax):
        # On calcule un angle sur dans la zone aveugle du parent
        angle_enfant = (self.angle + random.uniform(vision_angle, 360 - vision_angle)) % 360 # Modulo 360 car on veut garder l'angle entre 0 et 360
        angle_enfant_rad = math.radians(angle_enfant)
        
        # On utilise cet angle pour placer l'enfant à 15 pixels de distance dans une direction différente de la vision du parent
        distance_spawn = 15
        new_x = self.x + distance_spawn * math.cos(angle_enfant_rad)
        new_y = self.y + distance_spawn * math.sin(angle_enfant_rad)
        
        # On borne avec les limites de l'environnement
        new_x = max(0, min(new_x, self.env_width))
        new_y = max(0, min(new_y, self.env_height))
            
        # On crée un nouvelle agent enfant
        enfant = Agent(id, new_x, new_y, self.energy, self.rotate_deg, self.env_width, self.env_height, self.cost_rotate, self.cost_move, self.cost_eat, self.cost_neuron, self.cost_metabolism, self.digestion_rate, self.digestion_interval)
        # On divise par deux l'énergie de l'enfant et du parent
        enfant.energy /= 2
        self.energy /= 2
        # On copie le cerveau du parent dans l'enfant
        enfant.sbn.num_nodes = self.sbn.num_nodes
        enfant.sbn.states = self.sbn.states.copy()
        enfant.sbn.weights = self.sbn.weights.copy()
        enfant.sbn.true_ids = self.sbn.true_ids.copy()
        enfant.sbn.next_historical_id = self.sbn.next_historical_id
        
        # On donne à l'enfant son angle de vision
        enfant.angle = angle_enfant
        
        # On fais évoluer l'enfant a la naissance
        enfant.sbn.mutation(pd, pi, wmax)
        
        return enfant
        
    def update(self, oeil, pd, pi, pw, wmax):
        # Si l'agent est mort on ne fais rien
        if not self.alive:
            return
        
        self.energy -= self.cost_metabolism
        
        # On appelle la mutation
        #self.sbn.mutation(pd, pi, wmax)
        self.sbn.evolution(pw, wmax)
        
        # On récupère les actions ordonnées par le cerveau
        action_eat, action_move, action_rotate = self.sbn.step(oeil)

        # On retire pour chaque neurone activé son coût
        self.energy -= np.sum(self.sbn.states) * self.cost_neuron
        
        # On effectue chaque action en vérifiant l'énergie et en la diminuant en fonction
        if action_rotate and self.energy >= self.cost_rotate:
            self.rotate()
            self.energy -= self.cost_rotate
        if action_move and self.energy >= self.cost_move: 
            self.move()
            self.energy -= self.cost_move
        if action_eat:
            if self.energy >= self.cost_eat:
                self.energy -= self.cost_eat
            else:
                action_eat = 0
        
        # Si l'agent n'a plus d'énergie il est donc mort
        if self.energy < 1:
            self.alive = False
            
        # On incremente le compteur de pas
        self.step += 1
        
        # On retourne l'action manger pour que l'environnement puisse le savoir
        return action_eat
    
    def sense(self, voisins, dist_vision_sq, dist_manger_sq, angle_vision):
        self.vision_input = 0
        self.proie_potentielle = None
        angle_regard = math.radians(self.angle)
        angle_vision_rad = math.radians(angle_vision)
        
        for other in voisins:
            if other is self or not other.alive:
                continue
            
            # On calcul la distance entre les deux agents
            dx = other.x - self.x
            dy = other.y - self.y
            dist_sq = dx*dx + dy*dy
            
            # On regarde si dans les voisins de l'agent, il y'en a un dans son cone de vision
            if dist_sq < dist_vision_sq:
                angle_cible = math.atan2(dy, dx)  
                diff = (angle_cible - angle_regard + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < angle_vision_rad:
                    self.vision_input = 1
                    # On vérifie qu'il est a bonne distance
                    if dist_sq < dist_manger_sq:
                        self.proie_potentielle = other
                        break # On arrête à la première proie vue et mangeable
    
    def draw(self, screen, overlay, size, vision_dist, fov, max_energy, offset_y, vision_cone, tracking=False):
        if not self.alive:
            return
        
        # On récupère la position de l'agent
        pos = (int(self.x), int(self.y + offset_y)) # offset crée le décalage pour laisser de la place pour le dashboard du haut
        
        # Affichage du cercle colorée
        ratio = max(0, min(self.energy / max_energy, 1))
        color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
        if tracking: # Si l'agent est ciblé
            # Halo de ciblage (cercle vide plus grand)
            pygame.draw.circle(screen, (255, 255, 255), pos, size + 10, 2)
            
        pygame.draw.circle(screen, color, pos, size)
        
        if vision_cone:
            # Affichage du cône de vision
            left_rad = math.radians(self.angle - fov)
            right_rad = math.radians(self.angle + fov)
            
            p_left = (self.x + vision_dist * math.cos(left_rad), (self.y + offset_y) + vision_dist * math.sin(left_rad))
            p_right = (self.x + vision_dist * math.cos(right_rad), (self.y + offset_y) + vision_dist * math.sin(right_rad))
            
            cone_color = (255, 0, 0, 40) if self.vision_input == 1 else (255, 255, 255, 20)
            pygame.draw.polygon(overlay, cone_color, [pos, p_left, p_right])

