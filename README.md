# SBN Evolution Simulator

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview
The **SBN (Sign Boolean Network) Evolution Simulator** is an agent-based evolutionary simulation framework designed for scientific research in artificial life, neuroevolution, and population dynamics. 

It simulates a population of autonomous agents equipped with a Sign Boolean Network that evolve over time through natural selection. Agents must manage their energy (metabolism), interact with their environment (foraging, photosynthesis), and reproduce. The simulator tracks the emergence of complex behaviors, spatial clustering (tribes), and evolutionary convergence.

This codebase is optimized for both visual interactive simulation and high-performance **headless** execution on compute clusters, making it suitable for generating large-scale datasets for scientific publications.

---

## 🧬 Key Features
- **Neuroevolution**: Agents are driven by a dynamic Sign Boolean Network (SBN). Neural topologies (nodes/synapses) mutate (insertion, deletion) and weights evolve over generations.
- **Energy Metabolism**: Comprehensive energy management including movement costs, neuron activation costs, digestion mechanisms, and varied food sources (Photosynthesis, Ground Feeding, or Mixed).
- **Spatial Optimization**: Implements an optimized spatial grid hashing system to ensure $O(1)$ average-case neighbor detection, allowing simulations of thousands of agents.
- **Headless Mode**: Fully decoupled UI. The simulation can run entirely without `pygame` or a graphical server (`X11`), which is critical for SLURM/HPC cluster deployments.
- **Data Export & Analysis**: Decoupled architecture where the simulation engine exports neural topologies and spatial data to CSV files. A dedicated suite of analysis scripts processes this data post-simulation.

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/SBN-Evolution.git
   cd SBN-Evolution
   ```

2. **Set up a virtual environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   You can install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
   
   *Contents of `requirements.txt`:*
   ```txt
   pygame==2.5.2
   numpy==1.26.4
   pandas==2.2.2
   matplotlib==3.8.4
   seaborn==0.13.2
   networkx==3.3
   scikit-learn==1.4.2
   ```

---

## 🚀 Usage

### 1. Running the Simulation
All simulation parameters (e.g., mutation rates, energy costs, population size) are centralized in the `PARAMS` dictionary inside `Environment.py`.

To launch the simulation:
```bash
python Environment.py
```

**Keyboard Controls (if GUI is enabled)**:
- `SPACE`: Pause / Resume simulation.
- `G`: Toggle rendering (disabling rendering accelerates computation while keeping the Pygame window open).
- `V`: Toggle agent vision cones.
- `S`: Save a summary of the current simulation statistics.
- `Left Click` (on an agent): View its Spiking Brain Network topology in real-time.

### 2. Running in Headless Mode (HPC / Clusters)
For massive data generation on servers without displays:
1. Open `Environment.py`.
2. Set `"HEADLESS_MODE": True` in the `PARAMS` dictionary.
3. Run the script normally (`python Environment.py`). The simulation will output progress to the console and save results in the `results/` folder without initializing Pygame.

---

## 📊 Post-Simulation Analysis

The simulator exports data (CSV files containing agents' spatial and neural data) into the `results/<TEST_NAME>/exports` folder. You can run various analyses on these datasets.

**Global Evolutionary Dynamics**
Generates plots showing the evolution of synaptic weights and spatial maps.
```bash
python Analysis.py --path results/Your_Test_Name
```

**Spatial Clustering (DBSCAN / KNN)**
Analyzes the formation of tribes and spatial scattering across different experimental conditions.
```bash
python AnalysisCluster.py --path results
```

**Convergence Speed**
Evaluates behavioral instability and evolutionary convergence over time.
```bash
python AnalysisConvergence.py --path results
```

---

## 📁 Repository Structure
- **`Environment.py`**: The core simulation loop and physics engine.
- **`Agent.py`**: The biological entity containing metabolism logic, reproduction, and interactions.
- **`SbNetwork.py`**: The Sign Boolean Network matrix operations and mutation algorithms.
- **`SpatialGrid.py`**: Spatial hashing logic for optimized collision and vision detection.
- **`Food.py`**: Passive energy sources in the environment.
- **`Renderer.py` & `Interface.py`**: Decoupled Pygame rendering and UI dashboards.
- **`GraphVisualization.py`**: NetworkX logic to plot neural topologies.
- **`ExportData.py`**: Handles CSV exports of simulation state.
- **`Analysis*.py`**: Suite of Python scripts for generating publication-ready plots.

---

## 📝 Citation
If you use this simulator in your scientific research, please cite the associated article:

```bibtex
@article{sbnevolution2026,
  title={Artificial evolution of signed boolean networks for autonomous agents},
  author={GAIDI Yanis, ACHOCH Hajar},
  year={2026}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
