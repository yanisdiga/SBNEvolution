import pygame
class Food:
    def __init__(self, x, y, energy=50):
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True
        
    def draw(self, screen, offset_y):
        if self.alive:
            # Cercle de couleur jaune pour la nouritture
            pygame.draw.circle(screen, (255, 0, 255), (int(self.x), int(self.y + offset_y)), 4)