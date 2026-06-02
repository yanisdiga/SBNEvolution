import random
import math
import numpy as np
from SbNetwork import SbNetwork

class Agent:
    """
    Represents an autonomous agent in the environment.
    """
    def __init__(self, id: int, x: float, y: float, energy: float, rotate_deg: float, env_width: float, env_height: float, cost_rotate: float, cost_move: float, cost_eat: float, cost_neuron: float, cost_metabolism: float):
        self.id = id
        self.x = x
        self.y = y
        self.energy = energy
        self.rotate_deg = rotate_deg
        self.env_width = env_width
        self.env_height = env_height
        self.sbn = SbNetwork()
        self.speed = 2.0
        self.angle = random.uniform(0, 360)
        
        # Variables storing simulation state
        self.alive = True
        self.target_prey = None
        self.vision_input = 0
        self.step_count = 0
        
        # Action costs
        self.cost_rotate = cost_rotate
        self.cost_move = cost_move
        self.cost_eat = cost_eat
        self.cost_neuron = cost_neuron
        self.cost_metabolism = cost_metabolism
        
        # Digestion
        self.stomach = 0.0
        
    def move(self) -> None:
        """Moves the agent forward based on its current angle and speed."""
        # Convert angle to radians for calculations
        rad = math.radians(self.angle)
    
        dx = self.speed * math.cos(rad)
        dy = self.speed * math.sin(rad)
        
        # Create new positions
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Update positions taking into account environment boundaries
        self.x = max(0, min(new_x, self.env_width))
        self.y = max(0, min(new_y, self.env_height))
        
    def rotate(self) -> None:
        """Rotates the agent by rotate_deg degrees."""
        self.angle += self.rotate_deg
        self.angle %= 360 # Keep angle between 0 and 360
        
    def eat(self, victim: 'Agent | Food') -> None:
        """Eats a victim (Food or another Agent), transferring energy."""
        victim.alive = False
        self.energy += victim.energy
        victim.energy = 0
    
    def digestion(self) -> float:
        """Empties the stomach and returns the waste amount."""
        waste = self.stomach
        self.stomach = 0
        return waste
    
    def division(self, new_id: int, vision_angle: float, prob_del: float, prob_ins: float, weight_max: int) -> 'Agent':
        """
        Creates a new child agent by division, splitting energy and inheriting the neural network.
        The child spawns in the blind spot of the parent.
        """
        # Calculate a safe angle in the parent's blind zone
        angle_child = (self.angle + random.uniform(vision_angle, 360 - vision_angle)) % 360
        angle_child_rad = math.radians(angle_child)
        
        # Use this angle to place the child at a distance in a direction outside parent's vision
        distance_spawn = 15.0
        new_x = self.x + distance_spawn * math.cos(angle_child_rad)
        new_y = self.y + distance_spawn * math.sin(angle_child_rad)
        
        # Constrain to environment boundaries
        new_x = max(0, min(new_x, self.env_width))
        new_y = max(0, min(new_y, self.env_height))
            
        # Create a new child agent
        child = Agent(new_id, new_x, new_y, self.energy, self.rotate_deg, self.env_width, self.env_height, self.cost_rotate, self.cost_move, self.cost_eat, self.cost_neuron, self.cost_metabolism)
        
        # Halve energy for both child and parent
        child.energy /= 2.0
        self.energy /= 2.0
        
        # Copy the parent's brain to the child
        child.sbn.num_nodes = self.sbn.num_nodes
        child.sbn.states = self.sbn.states.copy()
        child.sbn.weights = self.sbn.weights.copy()
        child.sbn.true_ids = self.sbn.true_ids.copy()
        child.sbn.next_historical_id = self.sbn.next_historical_id
        
        # Set child's initial angle
        child.angle = angle_child
        
        # Mutate the child at birth
        child.sbn.mutation(prob_del, prob_ins, weight_max)
        
        return child
        
    def update(self, eye_input: int, prob_del: float, prob_ins: float, prob_weight: float, weight_max: int) -> int:
        """
        Updates the agent's state, runs the neural network step, and performs actions.
        Returns the eat action (0 or 1).
        """
        # If the agent is dead, do nothing
        if not self.alive:
            return 0
        
        self.energy -= self.cost_metabolism
        lost_energy = self.cost_metabolism
        
        # Call neural network evolution (weight update)
        self.sbn.evolution(prob_weight, weight_max)
        
        # Get actions ordered by the brain
        action_eat, action_move, action_rotate = self.sbn.step(eye_input)

        # Deduct cost for each activated neuron
        value_cost_neuron = np.sum(self.sbn.states) * self.cost_neuron
        lost_energy += value_cost_neuron
        self.energy -= value_cost_neuron
        
        # Perform each action checking energy limits and decreasing it accordingly
        if action_rotate and self.energy >= self.cost_rotate:
            self.rotate()
            self.energy -= self.cost_rotate
            lost_energy += self.cost_rotate
        if action_move and self.energy >= self.cost_move: 
            self.move()
            self.energy -= self.cost_move
            lost_energy += self.cost_move
        if action_eat:
            if self.energy >= self.cost_eat:
                self.energy -= self.cost_eat
                lost_energy += self.cost_eat
            else:
                action_eat = 0
        
        # If agent has no energy left, it dies
        if self.energy <= 0:
            self.alive = False
            
        # Increment step counter
        self.step_count += 1
        
        # Accumulate spent energy in the stomach
        self.stomach += lost_energy
        
        # Return eat action so the environment knows
        return action_eat
    
    def sense(self, neighbors: list, dist_vision_sq: float, dist_eat_sq: float, angle_vision: float) -> None:
        """
        Senses the environment based on neighbors. 
        Updates vision_input and sets target_prey if one is in range and view.
        """
        self.vision_input = 0
        self.target_prey = None
        angle_view = math.radians(self.angle)
        angle_vision_rad = math.radians(angle_vision)
        
        for other in neighbors:
            if other is self or not other.alive:
                continue
            
            # Calculate squared distance between agents
            dx = other.x - self.x
            dy = other.y - self.y
            dist_sq = dx*dx + dy*dy
            
            # Check if neighbor is within vision distance
            if dist_sq < dist_vision_sq:
                angle_target = math.atan2(dy, dx)  
                diff = (angle_target - angle_view + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < angle_vision_rad:
                    self.vision_input = 1
                    # Check if it's within eating distance
                    if dist_sq < dist_eat_sq:
                        self.target_prey = other
                        break # Stop at the first visible and edible prey
