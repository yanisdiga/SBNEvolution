def update_grid(agents: list, foods: list, cell_size: float, food_mode: int) -> dict:
    """
    Groups entities into a spatial grid based on cell_size to optimize neighbor search.
    """
    grid = {}
    
    # Create a global list containing agents
    entities = agents.copy()
    
    # If in feeding mode, add active food to the list
    if food_mode == 2 or food_mode == 3:
        # Only add food whose 'active' attribute is True (edible)
        entities.extend([f for f in foods if f.active])
    
    for entity in entities:
        # Calculate cell index
        cx = int(entity.x // cell_size)
        cy = int(entity.y // cell_size)
        
        cell_key = (cx, cy)
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(entity)
        
    return grid

def get_neighbors(agent, grid: dict, cell_size: float) -> list:
    """
    Retrieves all entities in the agent's cell and the 8 surrounding cells.
    """
    neighbors = []
    cx = int(agent.x // cell_size)
    cy = int(agent.y // cell_size)

    # Loop over the 9 cells (agent's cell + 8 neighbors)
    for i in range(cx - 1, cx + 2):
        for j in range(cy - 1, cy + 2):
            if (i, j) in grid:
                neighbors.extend(grid[(i, j)])
    
    return neighbors