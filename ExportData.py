import numpy as np
import csv
import os

def save_nodes_influence(pop: list, iteration: int, test_name: str, max_depth: int = 5):
    """
    Saves the influence matrix of each agent's neural network to a CSV file.
    """
    folder = os.path.join(test_name, "exports")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    header = [
        "Agent_ID", "Pos_X", "Pos_Y",
        "Eye_Forward", "Eye_Rotate",
        "Forward_Rotate", "Rotate_Forward",
        "Forward_Eye", "Rotate_Eye"
    ]
    
    file_name = os.path.join(folder, f"agents_save_{iteration}.csv")
    
    with open(file_name, mode="w", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header)
        
        for agent in pop:
            W = agent.sbn.weights
        
            # Initialize influence matrix as float
            I = np.zeros_like(W, dtype=float)
            
            # Matrix to store successive powers
            W_power = np.eye(agent.sbn.num_nodes)
            
            # Calculate influence using fixed depth
            for k in range(1, max_depth + 1):
                W_power = np.matmul(W_power, W)
                I += W_power
            
            # Extract spatial and identification data
            agent_id = agent.id
            pos_x = agent.x
            pos_y = agent.y
            
            # Extraction with real indices:
            # 0=Eye, 1=Mouth, 2=Fin F (Forward), 3=Fin R (Rotate)
            e_f = I[0, 2]  # Eye -> Forward
            e_r = I[0, 3]  # Eye -> Rotate
            f_r = I[2, 3]  # Forward -> Rotate
            r_f = I[3, 2]  # Rotate -> Forward
            f_e = I[2, 0]  # Forward -> Eye
            r_e = I[3, 0]  # Rotate -> Eye
            
            # Create and write row
            agent_row = [
                agent_id, pos_x, pos_y,
                e_f, e_r, f_r, r_f, f_e, r_e
            ]
            writer.writerow(agent_row)
            
    print(f"Brain save generated at iteration: {iteration} in file {file_name}")