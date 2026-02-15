import pygame
import random
import math
from Agent import Agent

WIDTH, HEIGHT = 1280, 720
N_ITER = 1
NUM_AGENTS = 10
BASE_ENERGY = 1000
ROTATE_DEG = 10

# --- INITIALISATION PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

agents = [Agent(i, random.randint(0, WIDTH), random.randint(0, HEIGHT), BASE_ENERGY, ROTATE_DEG, WIDTH, HEIGHT) for i in range(NUM_AGENTS)]

running = True
show_graphics = True
total_steps = 0

while running:
    # 1. INDISPENSABLE : Dire à l'OS que le programme répond
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # 2. LOGIQUE
    for _ in range(N_ITER):
        total_steps += 1
        
        for agent in agents:
            if not agent.alive:
                continue
            
            input_oeil = 0 # Pour l'instant, on met 0 car tu n'as pas encore codé la nourriture
            agent.update(input_oeil)
            
        agents = [a for a in agents if a.alive]
        
        
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
                pygame.draw.circle(screen, color, pos, 5)
                
                # Pour voir la direction ou les agents regarde
                end_x = agent.x + 10 * math.cos(agent.angle)
                end_y = agent.y + 10 * math.sin(agent.angle)
                pygame.draw.line(screen, (255, 255, 255), pos, (end_x, end_y), 1)
        
        # On rafraîchit l'écran une fois après la boucle des agents
        pygame.display.flip()
    
    clock.tick(60)

pygame.quit()