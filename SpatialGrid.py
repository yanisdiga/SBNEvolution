# On applique ces méthode car boucler sur tout les agents alors que certain sont très loin nous consommaient trop de ressources (bcp trop !)
def update_grid(agents, CELL_SIZE):
    grid = {}
    for agent in agents:
        # Calcul de l'index de la case
        cx = int(agent.x // CELL_SIZE)
        cy = int(agent.y // CELL_SIZE)
        
        cell_key = (cx, cy)
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(agent)
    return grid

def get_neighbors(agent, grid, CELL_SIZE):
    neighbors = []
    cx = int(agent.x // CELL_SIZE)
    cy = int(agent.y // CELL_SIZE)

    # On boucle sur les 9 cases (celle de l'agent + les 8 voisines)
    for i in range(cx - 1, cx + 2):
        for j in range(cy - 1, cy + 2):
            if (i, j) in grid:
                neighbors.extend(grid[(i, j)])
    
    return neighbors