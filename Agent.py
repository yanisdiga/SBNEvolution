import random
import math
import pygame
from SbNetwork import SbNetwork

class Agent:
    def __init__(self, id, x, y, energy, rotate_deg, env_width, env_height):
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
        
        # Cout des actions
        self.cost_rotate = 1
        self.cost_move = 1
        self.cost_eat = 1
        
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
    
    def division(self, id, x, y):
        # On crée un nouvelle agent enfant
        enfant = Agent(id, x, y, self.energy, self.rotate_deg, self.env_width, self.env_height)
        # On divise par deux l'énergie de l'enfant et du parent
        enfant.energy /= 2
        self.energy /= 2
        # On copie le cerveau du parent dans l'enfant
        enfant.sbn.num_nodes = self.sbn.num_nodes
        enfant.sbn.states = self.sbn.states.copy()
        enfant.sbn.weights = self.sbn.weights.copy()
        
        # On fais en sorte que l'enfant regarde a l'inverse du parent pour ne pas se faire manger a la naissance
        enfant.angle = self.angle + 180
        enfant.angle %= 360 # On garde l'angle entre 0 et 360
        
        return enfant
        
    def update(self, oeil, pd, pi, wmax):
        # Si l'agent est mort on ne fais rien
        if not self.alive:
            return
        
        # On appelle la mutation
        self.sbn.mutation(pd, pi, wmax)
        
        # On récupère les actions ordonnées par le cerveau
        action_eat, action_move, action_rotate = self.sbn.step(oeil)
        
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

        # On retourne l'action manger pour que l'environnement puisse le savoir
        return action_eat
    
    
    def sense(self, voisins, dist_vision_sq, dist_manger_sq, angle_vision):
        self.vision_input = 0
        self.proie_potentielle = None
        
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
                angle_regard = math.radians(self.angle)
                
                diff = (angle_cible - angle_regard + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < math.radians(angle_vision):
                    self.vision_input = 1
                    # On vérifie qu'il est a bonne distance
                    if dist_sq < dist_manger_sq:
                        self.proie_potentielle = other
                    break # On arrête à la première proie vue
                
                
                
    def draw(self, screen, overlay, size, vision_dist, fov, max_energy):
        if not self.alive:
            return
        
        # On récupère la position de l'agent
        pos = (int(self.x), int(self.y))
        
        # Affichage du cercle colorée
        ratio = max(0, min(self.energy / max_energy, 1))
        color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
        pygame.draw.circle(screen, color, pos, size)
        
        # Affichage du cône de vision
        left_rad = math.radians(self.angle - fov)
        right_rad = math.radians(self.angle + fov)
        
        p_left = (self.x + vision_dist * math.cos(left_rad), self.y + vision_dist * math.sin(left_rad))
        p_right = (self.x + vision_dist * math.cos(right_rad), self.y + vision_dist * math.sin(right_rad))
        
        cone_color = (255, 0, 0, 40) if self.vision_input == 1 else (255, 255, 255, 20)
        pygame.draw.polygon(overlay, cone_color, [pos, p_left, p_right])