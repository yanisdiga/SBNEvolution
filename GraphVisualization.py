import networkx as nx
import matplotlib.pyplot as plt
import os

def show_sbn_graph(agent_id: int, sbn):
    """
    Displays the neural network graph of an agent.
    """
    plt.figure(figsize=(18, 8))
    plt.suptitle(f"Brain of agent {agent_id} (Nodes: {sbn.num_nodes})", fontsize=16)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Translation dictionary
    name_nodes = {0: "E", 1: "M", 2: "F", 3: "R", 4: "R1", 5: "R2", 6: "R3", 7: "R4"}
    
    # Add generic names (N8, N9...) if the network mutated and grew
    for i in range(8, sbn.num_nodes):
        name_nodes[i] = f"N{i}"

    # Create each node from top to bottom based on function (for readability)
    for id in sbn.true_ids:
        name = name_nodes.get(id, f"N{id}")
        if id in [0, 1]:         # Eye, Mouth
            layer = 3           # Top layer
        elif id in [4, 5, 6]:    # R1, R2, R3
            layer = 1           # Bottom layer
        elif id in [3, 7]:       # R, R4
            layer = 0           # Bottom layer
        else:                   # F and new nodes (N8, N9...)
            layer = 2           # Middle layer
            
        G.add_node(name, layer=layer)

    # Traverse weight matrix
    edges_green = []
    edges_red = []
    
    for i in range(sbn.num_nodes):
        for j in range(sbn.num_nodes):
            weight = sbn.weights[i, j]
            if weight != 0:
                id_start = sbn.true_ids[i]
                id_end = sbn.true_ids[j]
                
                # Generate name dynamically if it's new (>= 8)
                name_start = name_nodes.get(id_start, f"N{id_start}")
                name_end = name_nodes.get(id_end, f"N{id_end}")
                
                # Add connection with its weight
                G.add_edge(name_start, name_end, weight=weight)
                
                # Sort for color display
                if weight > 0:
                    edges_green.append((name_start, name_end))
                else:
                    edges_red.append((name_start, name_end))

    # align="horizontal" forces layers to be on horizontal lines
    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")

    bottom_nodes = ["R", "R1", "R2", "R3", "R4"]
    # Get all X coordinates of the bottom layer and sort them
    x_coords_bottom = sorted([pos[n][0] for n in bottom_nodes if n in pos])
    
    # Reassign these sorted X coordinates to our nodes in the correct order
    current_idx = 0
    for node in bottom_nodes:
        if node in pos:
            pos[node] = (x_coords_bottom[current_idx], pos[node][1])
            current_idx += 1
    
    # Align E, F and R vertically on the left
    # Take the leftmost X coordinate (which now belongs to R)
    x_left = pos["R"][0]
    pos["E"] = (x_left, pos["E"][1])
    pos["F"] = (x_left, pos["F"][1])
    
    # Dynamic calculation of node sizes, edges, and text
    # Scale factor: 1.0 if there are 8 nodes, 0.5 if there are 16, etc.
    scale_factor = 8 / max(8, sbn.num_nodes)
    
    # Define minimum limits so it remains visible on screen
    node_size = max(500, int(2000 * scale_factor))
    font_size = max(6, int(10 * scale_factor))
    arrow_size = max(10, int(20 * scale_factor))
    
    # 1. Draw nodes with dynamic size
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold")
    
    # 2. Draw edges (arrows)
    nx.draw_networkx_edges(G, pos, edgelist=edges_green, edge_color='green', 
                           arrows=True, arrowsize=arrow_size, width=2,
                           node_size=node_size)
                           
    nx.draw_networkx_edges(G, pos, edgelist=edges_red, edge_color='red', 
                           arrows=True, arrowsize=arrow_size, width=2,
                           node_size=node_size)
    
    # Get weights of all edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    # Draw all labels manually
    for (u, v), weight in edge_labels.items():
        weight_text = f"{weight:.2f}"
        
        if u == v:
            # Self-loop: shift text above the node
            x, y = pos[u]
            plt.text(x, y + 0.15, weight_text, fontsize=8, color='black', 
                     ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        else:
            # Normal connection: calculate perfect geometric center
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            
            # Shift text based on alphabetical order to prevent overlapping labels
            if str(u) < str(v):
                x_mid += 0.02
                y_mid += 0.04
            else:
                x_mid -= 0.02
                y_mid -= 0.04
                
            plt.text(x_mid, y_mid, weight_text, fontsize=8, color='black', 
                     ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Show window
    plt.show()

def show_simulation_summary(history_steps: list, history_pop: list, history_size: list, history_energy: list, history_nodes_activated: list, history_global_energy: list, save_path: str = None):
    """
    Displays global simulation statistics.
    """
    plt.figure(figsize=(12, 8))
    plt.suptitle("Simulation Summary", fontsize=16)

    # --- GRAPH 1: Population Evolution ---
    plt.subplot(3, 2, 1)
    plt.plot(history_steps, history_pop, color='green')
    plt.title("Total Population")
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of agents")
    plt.grid(True, alpha=0.3)

    # --- GRAPH 2: Average Brain Size (SBN) ---
    plt.subplot(3, 2, 2)
    plt.plot(history_steps, history_size, color='blue')
    plt.title("Average network size (Nodes)")
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of neurons")
    plt.grid(True, alpha=0.3)

    # --- GRAPH 3: Average Agent Energy ---
    plt.subplot(3, 2, 3)
    plt.plot(history_steps, history_energy, color='orange')
    plt.title("Average energy")
    plt.xlabel("Number of iterations")
    plt.ylabel("Energy units")
    plt.grid(True, alpha=0.3)

    # --- GRAPH 4: Average Activated Nodes per Agent ---
    plt.subplot(3, 2, 4)
    plt.plot(history_steps, history_nodes_activated, color='red')
    plt.title("Average activated neurons")
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of neurons")
    plt.grid(True, alpha=0.3)
    
    # --- GRAPH 5: Global System Energy ---
    plt.subplot(3, 2, 5)
    plt.plot(history_steps, history_global_energy, color='purple')
    plt.title("Global system energy")
    plt.xlabel("Number of iterations")
    plt.ylabel("Available energy")
    if history_global_energy:
        initial_val = history_global_energy[0]
        plt.ylim(initial_val - 10, initial_val + 10)
    plt.ticklabel_format(useOffset=False, style='plain', axis='y')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
