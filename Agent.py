import random
import math
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
        
        self.alive = True
        
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