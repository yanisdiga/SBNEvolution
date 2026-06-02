import numpy as np
import random

class SbNetwork:
    """
    Spiking Brain Network (SBN) - Neural network controlling the agent.
    """
    def __init__(self):
        self.num_nodes = 8
        # State matrix creation
        self.states = np.zeros(self.num_nodes, dtype=int)
        
        # Node meanings (Indices)
        # 0. Eye
        # 1. Mouth
        # 2. Fin F (Forward)
        # 3. Fin R (Rotate)
        # 4. R1
        # 5. R2
        # 6. R3
        # 7. R4
        
        # Weight matrix creation
        self.weights = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        
        # Initialize states
        self.states[2] = 1 # Fin F is always activated initially
        self.states[3] = 1 # Fin R is activated initially
        
        # Initialize weights
        self.weights[0, 1] = 1 # Eye activates Mouth
        self.weights[2, 2] = 1 # Fin F self-activates
        self.weights[3, 3] = -1 # R self-inhibits
        self.weights[3, 4] = 1 # R activates R1 ...
        self.weights[4, 4] = -1
        self.weights[4, 5] = 1
        self.weights[5, 5] = -1
        self.weights[5, 6] = 1
        self.weights[6, 6] = -1
        self.weights[6, 7] = 1
        self.weights[7, 7] = -1
        self.weights[7, 3] = 1
        
        # Store true IDs of each node (for visualization)
        self.true_ids = list(range(self.num_nodes)) # [0, 1, 2, 3, 4, 5, 6, 7]
        self.next_historical_id = 8 # Next created neuron will be called N8
        
    def step(self, eye_input: int):
        # Update eye state based on environment
        self.states[0] = eye_input
        
        # Calculate weighted sum for all nodes at once (Matrix product)
        x = self.states @ self.weights
        
        # Apply threshold (if x > 0 -> 1, else 0)
        new_states = (x > 0).astype(int)
        
        # Restore correct eye value
        new_states[0] = eye_input
        
        # Synchronous update
        self.states = new_states
        
        # Return actuator states: Mouth(Eat), Fin F(Move), Fin R(Rotate)
        return self.states[1], self.states[2], self.states[3]

    def mutation(self, prob_del: float, prob_ins: float, weight_max: int):
        if random.random() < prob_del:
            self.deletion()
        if random.random() < prob_ins:
            self.insertion(weight_max)
    
    def evolution(self, prob_weight: float, weight_max: int):
        if random.random() < prob_weight:
            self.weight_update(weight_max)
    
    def weight_update(self, weight_max: int):
        entry_node = random.randint(0, self.num_nodes - 1)
        exit_node = random.randint(0, self.num_nodes - 1)
        self.weights[entry_node, exit_node] += random.randint(-1, 1)
        self.weights[entry_node, exit_node] = np.clip(self.weights[entry_node, exit_node], -weight_max, weight_max)
            
    def insertion(self, weight_max: int):
        n = self.num_nodes
        # Add a node to the neural network
        self.states = np.append(self.states, random.randint(0, 1))
        # Create a new weight matrix one size larger
        new_weights = np.zeros((n + 1, n + 1), dtype=int)
        # Copy the old matrix into the new one
        new_weights[:n, :n] = self.weights
        
        # Relay method for initialization
        entry_node = random.randint(0, self.num_nodes - 1)
        exit_node = random.randint(0, self.num_nodes - 1)
        new_weights[entry_node, n] = random.randint(-weight_max, weight_max)
        new_weights[n, exit_node] = random.randint(-weight_max, weight_max)
        
        # Replace old weight matrix
        self.weights = new_weights
        self.num_nodes += 1 # Update node count
        self.true_ids.append(self.next_historical_id) # Add new node to ID list
        self.next_historical_id += 1 # Update next historical ID
        
    def deletion(self):
        # Prevent deleting base neurons
        if self.num_nodes <= 4:
            return
        
        node_to_delete = random.randint(4, self.num_nodes - 1)
        self.states = np.delete(self.states, node_to_delete)
        self.weights = np.delete(self.weights, node_to_delete, axis=0)
        self.weights = np.delete(self.weights, node_to_delete, axis=1)
        self.num_nodes -= 1
        self.true_ids.pop(node_to_delete)