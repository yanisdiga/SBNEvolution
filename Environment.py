import random
import math
import json
import os
import numpy as np
from Agent import Agent
from GraphVisualization import show_sbn_graph, show_simulation_summary
from SpatialGrid import update_grid, get_neighbors
from Food import Food
from ExportData import save_nodes_influence
import sys

# ==========================================================
# SIMULATION CONFIGURATION
# ==========================================================

# Simulation parameters
PARAMS = {
    "TEST_NAME": "Weights_cap_seed",
    "SEED": 45,
    "NUM_AGENTS": 200,
    "BASE_ENERGY": 1000.0,
    "DIVISION_ENERGY": 3000.0,
    "PROB_DELETION": 0.05,
    "PROB_INSERTION": 0.05,
    "PROB_EVOLUTION": 0.05,
    "WEIGHT_MAX": 3,
    "DISTANCE_VISION": 70.0,
    "VISION_ANGLE": 30.0,
    "FOOD_MODE": 2,
    "COST_NEURON": 0.3,
    "COST_MOVE": 0.02,
    "COST_ROTATE": 0.1,  
    "COST_EAT": 0.01,   
    "COST_METABOLISM": 0.1,   
    "NUM_FOOD": 10,
    "FEEDING_BOOST": 500.0,
    "DIGESTION_INTERVAL": 120,
    "DIGESTION_MIN": 250.0,
    "HEADLESS_MODE": False,  # Set to True to disable Pygame entirely
}

# Fetch parameters from dictionary or use default values
# --- GENERAL PARAMETERS ---
TEST_NAME = PARAMS.get("TEST_NAME", "Initial_Extinction") 
SEED = PARAMS.get("SEED", 42)                            

# --- ENGINE AND DISPLAY ---
WIDTH = PARAMS.get("WIDTH", 1280)                        
HEIGHT = PARAMS.get("HEIGHT", 720)                       
N_ITER = PARAMS.get("N_ITER", 1)                         
FPS = PARAMS.get("FPS", 60)                              
DISPLAY_FPS = PARAMS.get("DISPLAY_FPS", 0)               
HEADLESS_MODE = PARAMS.get("HEADLESS_MODE", False)

# --- PHYSICS AND AGENT CAPABILITIES ---
AGENT_SIZE = PARAMS.get("AGENT_SIZE", 5)                        
ROTATE_DEG = PARAMS.get("ROTATE_DEG", 10.0)                           
VISION_ANGLE = PARAMS.get("VISION_ANGLE", 45.0)                       
DISTANCE_EAT = PARAMS.get("DISTANCE_EAT", AGENT_SIZE * 2)     
DISTANCE_VISION = PARAMS.get("DISTANCE_VISION", DISTANCE_EAT)    

# --- POPULATION DYNAMICS AND ENERGY ---
NUM_AGENTS = PARAMS.get("NUM_AGENTS", 1000)              
BASE_ENERGY = PARAMS.get("BASE_ENERGY", 100.0)             
DIVISION_ENERGY = PARAMS.get("DIVISION_ENERGY", 600.0)     
FOOD_MODE = PARAMS.get("FOOD_MODE", 1)                   # 1: Photosynthesis, 2: Ground food, 3: Mixed
PHOTOSYNTHESIS_INTERVAL = PARAMS.get("PHOTOSYNTHESIS_INTERVAL", 10)   
FEEDING_BOOST = PARAMS.get("FEEDING_BOOST", 1.0)          
PHOTOSYNTHESIS_BOOST = PARAMS.get("PHOTOSYNTHESIS_BOOST", 1.0)  
PHOTOSYNTHESIS_DECREASE = PARAMS.get("PHOTOSYNTHESIS_DECREASE", 0.01)
PHOTOSYNTHESIS_INTERVAL_UPDATE = PARAMS.get("PHOTOSYNTHESIS_INTERVAL_UPDATE", 100)
PHOTOSYNTHESIS_MIN = PARAMS.get("PHOTOSYNTHESIS_MIN", 0.0)
NUM_FOOD = PARAMS.get("NUM_FOOD", 100)
COST_MOVE = PARAMS.get("COST_MOVE", 1.0)
COST_ROTATE = PARAMS.get("COST_ROTATE", 1.0)
COST_EAT = PARAMS.get("COST_EAT", 1.0)
COST_NEURON = PARAMS.get("COST_NEURON", 1.0)
COST_METABOLISM = PARAMS.get("COST_METABOLISM", 0.05)
DIGESTION_MIN = PARAMS.get("DIGESTION_MIN", 40.0)
DIGESTION_RATE = PARAMS.get("DIGESTION_RATE", 5.0)
DIGESTION_INTERVAL = PARAMS.get("DIGESTION_INTERVAL", 10)
ACTIVE_CORPSE = PARAMS.get("ACTIVE_CORPSE", True)

# --- BRAIN MUTATIONS (NEURAL NETWORK) ---
PROB_DELETION = PARAMS.get("PROB_DELETION", 0.01)      
PROB_INSERTION = PARAMS.get("PROB_INSERTION", 0.02)    
PROB_EVOLUTION = PARAMS.get("PROB_EVOLUTION", 0.02)    
WEIGHT_MAX = PARAMS.get("WEIGHT_MAX", 3)     

# --- STATISTICS AND VISUALIZATION ---
TRACKING_ID = PARAMS.get("TRACKING_ID", None)
    
# Grid parameters
CELL_SIZE = DISTANCE_VISION * 1.2  # Must be >= DISTANCE_VISION (added 20% margin)

random.seed(SEED) 
np.random.seed(SEED) 

# Information Dashboard
DASHBOARD_SIZE = 50

# --- SIMULATION SAVE ---
# Creation of save folder and parameters save
RESULT_FOLDER = "results"
os.makedirs(RESULT_FOLDER, exist_ok=True)
BASE_TEST_NAME = TEST_NAME
SIMULATION_SAVE_FOLDER = os.path.join(RESULT_FOLDER, TEST_NAME)

if os.path.exists(SIMULATION_SAVE_FOLDER):
    count = 1
    # Find the first available number
    while os.path.exists(os.path.join(RESULT_FOLDER, f"{BASE_TEST_NAME}_{count}")):
        count += 1
    # Update final name
    TEST_NAME = f"{BASE_TEST_NAME}_{count}"
    SIMULATION_SAVE_FOLDER = os.path.join(RESULT_FOLDER, TEST_NAME)

PARAMS["TEST_NAME"] = TEST_NAME

os.makedirs(SIMULATION_SAVE_FOLDER)

# Save the dictionary in a parameters.json file
json_path = os.path.join(SIMULATION_SAVE_FOLDER, "parameters.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(PARAMS, f, indent=4)

# --- INITIALIZE UI IF NEEDED ---
if not HEADLESS_MODE:
    import pygame
    from Interface import draw_dashboard, show_graphics_off, graphics_pause
    from Renderer import Renderer
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + DASHBOARD_SIZE))
    overlay = pygame.Surface((WIDTH, HEIGHT + DASHBOARD_SIZE), pygame.SRCALPHA)
    pygame.display.set_caption("SBN Evolution Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

# List storing agents and food
agents = [Agent(i, random.randint(0, WIDTH), random.randint(0, HEIGHT), BASE_ENERGY, ROTATE_DEG, WIDTH, HEIGHT, COST_ROTATE, COST_MOVE, COST_EAT, COST_NEURON, COST_METABOLISM) for i in range(NUM_AGENTS)]
foods = [Food(random.randint(0, WIDTH), random.randint(0, HEIGHT), energy=FEEDING_BOOST) for _ in range(NUM_FOOD)]

running = True
show_graphics = not HEADLESS_MODE # Variable allowing to show or hide simulation
total_steps = 0 # Variable storing the number of simulation steps

new_id = NUM_AGENTS # New id to increment from the initial number of agents (for descendants)

is_paused = False # Variable allowing to pause the simulation
simulated_time_ms = 0

vision_cone = True

# Pre-calculate squared distance to avoid square roots (SLOW)
DIST_EAT_SQ = DISTANCE_EAT * DISTANCE_EAT
DISTANCE_VISION_SQ = DISTANCE_VISION**2

# Retrieve original photosynthesis value
PHOTOSYNTHESIS_INITIAL_BOOST = PHOTOSYNTHESIS_BOOST

# Lists to store history
stats_steps = []
stats_pop = []
stats_size = []
stats_energy = []
stats_node_activated = []
stats_global_energy = []

winter = False

digestion_calendar = {} # Calendar of waste to activate (queue)

while running:
    # 1. Input handling (Only if not headless)
    if not HEADLESS_MODE:
        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: # Space to Pause/Unpause
                    is_paused = not is_paused
                    if not show_graphics: show_graphics_off(screen, font, WIDTH, HEIGHT, is_paused, DASHBOARD_SIZE)
                elif event.key == pygame.K_g:   # G key to toggle rendering
                    show_graphics = not show_graphics     
                    if not show_graphics: show_graphics_off(screen, font, WIDTH, HEIGHT, is_paused, DASHBOARD_SIZE)
                elif event.key == pygame.K_v:   # V key to toggle vision cones
                    vision_cone = not vision_cone
                elif event.key == pygame.K_s:
                    show_simulation_summary(stats_steps, stats_pop, stats_size, stats_energy, stats_node_activated, stats_global_energy)
                        
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 corresponds to left click
                    is_paused = True
                    mx, my = pygame.mouse.get_pos()
                    # Bring mouse click into simulation reference frame
                    my = my - DASHBOARD_SIZE
                    
                    radius_sq = AGENT_SIZE * AGENT_SIZE
                    # Search which agent was clicked
                    for agent in agents:
                        if not agent.alive: continue
                        dx = mx - agent.x
                        dy = my - agent.y
                        
                        # Circle / Point collision test without square root
                        if (dx*dx + dy*dy) <= radius_sq:
                            # Call SBN graph visualization function
                            show_sbn_graph(agent.id, agent.sbn)
                            break
    
    # 2. LOGIC
    if not is_paused:
        for _ in range(N_ITER):
            total_steps += 1
            if not HEADLESS_MODE:
                simulated_time_ms += clock.get_time()
            else:
                simulated_time_ms += 16 # Approximation (60 fps = 16ms)
                
            new_children = []
            
            if total_steps % 50000 == 0:
                save_nodes_influence(agents, total_steps, SIMULATION_SAVE_FOLDER)
            
            # Waste activations
            if total_steps in digestion_calendar:
                for waste_to_activate in digestion_calendar[total_steps]:
                    waste_to_activate.active = True
                # Clear calendar memory for this turn
                del digestion_calendar[total_steps]
            
            # Shuffle for fairness
            random.shuffle(agents)
            
            spatial_grid = update_grid(agents, foods, CELL_SIZE, FOOD_MODE)
            
            eat_intentions = {} # Dictionary storing each agent's intention to eat

            if FOOD_MODE == 3:
                if total_steps % PHOTOSYNTHESIS_INTERVAL_UPDATE == 0:
                    if PHOTOSYNTHESIS_BOOST <= PHOTOSYNTHESIS_MIN:
                        winter = False
                    elif PHOTOSYNTHESIS_BOOST >= PHOTOSYNTHESIS_INITIAL_BOOST: 
                        winter = True
                    if winter:
                        PHOTOSYNTHESIS_BOOST = max(PHOTOSYNTHESIS_BOOST - PHOTOSYNTHESIS_DECREASE, PHOTOSYNTHESIS_MIN)
                    else:
                        PHOTOSYNTHESIS_BOOST = min(PHOTOSYNTHESIS_BOOST + PHOTOSYNTHESIS_DECREASE, PHOTOSYNTHESIS_INITIAL_BOOST)
            
            for agent in agents:
                if not agent.alive:
                    continue
                
                # Energy boost every N_BOOST time steps
                if FOOD_MODE == 1 or FOOD_MODE == 3:
                    if total_steps % PHOTOSYNTHESIS_INTERVAL == 0:
                        agent.energy += PHOTOSYNTHESIS_BOOST
                
                # Retrieve list of nearby cells
                neighbors = get_neighbors(agent, spatial_grid, CELL_SIZE)
                
                # Register in agent.target_prey if a target is in range
                agent.sense(neighbors, DISTANCE_VISION_SQ, DIST_EAT_SQ, VISION_ANGLE)
                
                # Eat decision
                action_eat = agent.update(agent.vision_input, PROB_DELETION, PROB_INSERTION, PROB_EVOLUTION, WEIGHT_MAX)
                
                # Eat
                # Use the prey found in the vision loop
                if action_eat and (agent.target_prey is not None):
                    victim = agent.target_prey
                    # Security check
                    if victim.alive:
                        if isinstance(victim, Agent):
                            eat_intentions[agent] = agent.target_prey
                        elif isinstance(victim, Food):
                            agent.eat(victim)
                            foods.remove(victim) # Remove food from ground
                            
                # Digestion
                if agent.stomach > DIGESTION_MIN:
                    waste = agent.digestion()
                    new_food = Food(agent.x, agent.y, energy=waste, active=False)
                    foods.append(new_food)
                    
                    target_step = total_steps + DIGESTION_INTERVAL
                    if target_step not in digestion_calendar:
                        digestion_calendar[target_step] = []
                    
                    # Add to calendar
                    digestion_calendar[target_step].append(new_food)
                    
                # Division
                if agent.energy >= DIVISION_ENERGY:
                    child = agent.division(new_id, VISION_ANGLE, PROB_DELETION, PROB_INSERTION, WEIGHT_MAX)
                    new_id += 1
                    new_children.append(child)
            
            # Conflict verification during the "eat" action
            for agent, victim in eat_intentions.items():
                # Verify that agent and victim are still alive
                if not agent.alive or not victim.alive: continue
                
                # CASE 1: Duel
                if eat_intentions.get(victim) == agent:
                    if agent.energy > victim.energy:
                        agent.eat(victim)
                    elif agent.energy == victim.energy:
                        if random.random() > 0.5: 
                            agent.eat(victim)
                        else: 
                            victim.eat(agent)
                    else:
                        victim.eat(agent)
                # CASE 2: Normal
                else:
                    agent.eat(victim)
            
            # Quick cleanup
            if ACTIVE_CORPSE:
                for a in agents:
                    # Target those who just died (starvation or old age) 
                    # and were not "emptied" by a predator
                    if not a.alive and a.stomach + a.energy != 0:
                        # Drop stomach contents on the ground
                        agent_rest = a.stomach + a.energy
                        new_food = Food(a.x, a.y, energy=agent_rest, active=False)
                        foods.append(new_food)
                        # Empty stomach to avoid duplicates if code loops over it
                        a.stomach = 0
                        
                        target_step = total_steps + DIGESTION_INTERVAL
                        if target_step not in digestion_calendar:
                            digestion_calendar[target_step] = []
                        
                        # Add to calendar
                        digestion_calendar[target_step].append(new_food)

            # Usual cleanup
            agents = [a for a in agents if a.alive]
            agents.extend(new_children)
            
            # Break simulation if all agents are dead to avoid infinite loop
            if len(agents) == 0 and HEADLESS_MODE:
                running = False
                break
        
    # 3. DISPLAY (RENDERING)
    if not HEADLESS_MODE and show_graphics:
        screen.fill((0, 0, 0))
        # Clear it every frame
        overlay.fill((0, 0, 0, 0))
        
        # Display agents
        for agent in agents:
            is_tracked = (agent.id == TRACKING_ID)
            if DISTANCE_VISION > 20: # Apply opacity on vision cone for better visibility
                Renderer.draw_agent(agent, screen, overlay, AGENT_SIZE, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY, DASHBOARD_SIZE, vision_cone=vision_cone, tracking=is_tracked) 
            else:
                Renderer.draw_agent(agent, screen, screen, AGENT_SIZE, DISTANCE_VISION, VISION_ANGLE, BASE_ENERGY, DASHBOARD_SIZE, vision_cone=vision_cone, tracking=is_tracked)
            
        if FOOD_MODE == 2 or FOOD_MODE == 3:
            for food in foods:
                Renderer.draw_food(food, screen, DASHBOARD_SIZE)
        
        # Merge layer with screen
        screen.blit(overlay, (0, 0))
        
        # Calculate average number of neurons
        pop = len(agents)
        if pop > 0:
            mean_nodes = sum(a.sbn.num_nodes for a in agents) / pop
        else:
            mean_nodes = 0.0
            
        # Display dashboard
        draw_dashboard(screen, clock, pop, total_steps, mean_nodes, PARAMS, WIDTH, DASHBOARD_SIZE, font, simulated_time_ms)
        
        if is_paused: graphics_pause(screen, font, DASHBOARD_SIZE)
        
        # Refresh screen once after agent loop
        pygame.display.flip()
        clock.tick(FPS)
        
    # 4. Statistics
    if total_steps % 100 == 0 and len(agents) > 0:
        stats_steps.append(total_steps)
        stats_pop.append(len(agents))
        agent_max = max(a.step_count for a in agents)
        age_max = max(a.step_count for a in agents)
        energy_agent = sum(a.energy + a.stomach for a in agents)
        energy_sol = sum(f.energy for f in foods)
        energy_total = energy_agent + energy_sol
        
        # Calculate averages
        avg_nodes = sum(a.sbn.num_nodes for a in agents) / len(agents)
        avg_energy = sum(a.energy for a in agents) / len(agents)
        avg_active_nodes = sum(sum(a.sbn.states) for a in agents) / len(agents)
        stats_size.append(avg_nodes)
        stats_energy.append(avg_energy)
        stats_node_activated.append(avg_active_nodes)
        stats_global_energy.append(energy_total)
        
        if HEADLESS_MODE or not show_graphics:
            elapsed_seconds = simulated_time_ms // 1000
            print(f"[{elapsed_seconds}s] Step: {total_steps} | Pop: {len(agents)} | Avg Energy: {avg_energy:.0f} | Avg Nodes: {avg_nodes:.1f} | Max Age: {age_max}")

if not HEADLESS_MODE:
    pygame.quit()

summary_save_folder = os.path.join(SIMULATION_SAVE_FOLDER, "simulation_summary.png")
show_simulation_summary(stats_steps, stats_pop, stats_size, stats_energy, stats_node_activated, stats_global_energy, summary_save_folder)
