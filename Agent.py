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
        
    def update(self, oeil):
        # Si l'agent est mort on ne fais rien
        if not self.alive:
            return
        
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
        if self.energy <= 0:
            self.alive = False

        # On retourne l'action manger pour que l'environnement puisse le savoir
        return action_eat