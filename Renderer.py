import pygame
import math
from Agent import Agent
from Food import Food

class Renderer:
    """
    Handles all Pygame drawing logic to keep the simulation core headless-ready.
    """
    @staticmethod
    def draw_agent(agent: Agent, screen, overlay, size: float, vision_dist: float, fov: float, max_energy: float, offset_y: float, vision_cone: bool, tracking: bool = False):
        if not agent.alive:
            return
        
        # Get agent position
        pos = (int(agent.x), int(agent.y + offset_y)) # offset leaves space for the top dashboard
        
        # Color based on energy
        ratio = max(0, min(agent.energy / max_energy, 1))
        color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
        
        if tracking: # If agent is tracked
            # Targeting halo
            pygame.draw.circle(screen, (255, 255, 255), pos, size + 10, 2)
            
        pygame.draw.circle(screen, color, pos, size)
        
        if vision_cone:
            # Display vision cone
            left_rad = math.radians(agent.angle - fov)
            right_rad = math.radians(agent.angle + fov)
            
            p_left = (agent.x + vision_dist * math.cos(left_rad), (agent.y + offset_y) + vision_dist * math.sin(left_rad))
            p_right = (agent.x + vision_dist * math.cos(right_rad), (agent.y + offset_y) + vision_dist * math.sin(right_rad))
            
            cone_color = (255, 0, 0, 40) if agent.vision_input == 1 else (255, 255, 255, 20)
            pygame.draw.polygon(overlay, cone_color, [pos, p_left, p_right])

    @staticmethod
    def draw_food(food: Food, screen, offset_y: float):
        if food.alive:
            if food.active:
                color = (255, 0, 255)
            else:
                color = (137, 81, 41)
            pygame.draw.circle(screen, color, (int(food.x), int(food.y + offset_y)), 4)
