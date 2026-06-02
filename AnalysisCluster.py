import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def spatial_analysis(parent_folder: str) -> pd.DataFrame:
    conditions = ["Bounded_Condition", "Unbounded_Condition"]
    global_data = []
    
    for condition in conditions:
        condition_path = os.path.join(parent_folder, condition)
        run_folders = glob.glob(os.path.join(condition_path, "Run_*"))
        
        if not run_folders:
            print(f"No run found in {condition_path}")
            continue
            
        for run_folder in run_folders:
            run_name = os.path.basename(run_folder)
            csv_files = glob.glob(os.path.join(run_folder, "exports", "*.csv"))
            
            for file in csv_files:
                file_name = os.path.basename(file)
                try:
                    iteration = int(file_name.split('_')[-1].replace('.csv', ''))
                except ValueError: 
                    continue
                
                # Open a single world
                df = pd.read_csv(file)
                if len(df) < 10: # Safety if species is almost extinct
                    continue
                    
                coordinates = df[['Pos_X', 'Pos_Y']].values
                
                # KNN Analysis (Scattering)
                knn = NearestNeighbors(n_neighbors=10)
                knn.fit(coordinates)
                distances, _ = knn.kneighbors(coordinates)
                scatter_score = np.mean(distances[:, 1:])
                
                # DBSCAN Analysis (Tribes)
                dbscan = DBSCAN(eps=50, min_samples=10)
                labels = dbscan.fit_predict(coordinates)
                
                num_tribes = len(set(labels)) - (1 if -1 in labels else 0)
                isolated_percentage = (list(labels).count(-1) / len(labels)) * 100
                
                # Save pure metric
                global_data.append({
                    "Condition": condition,
                    "Seed": run_name,
                    "Iteration": iteration,
                    "Scatter_Score": scatter_score,
                    "Num_Tribes": num_tribes,
                    "Isolated_Percentage": isolated_percentage
                })
                
    # Create Super-Table
    super_df = pd.DataFrame(global_data)
    if not super_df.empty:
        super_df = super_df.sort_values(by="Iteration")
    
    return super_df

def draw_spatial_graph(super_df: pd.DataFrame, parent_folder: str):
    sns.set_theme(style="whitegrid") 
    save_folder = os.path.join(parent_folder, "Macro_Spatial_Analysis")
    os.makedirs(save_folder, exist_ok=True)
    
    # 1. Scatter Score Graph
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=super_df, x="Iteration", y="Scatter_Score", hue="Condition", linewidth=2, errorbar=None)
    plt.title("Spatial Scattering Dynamics (Higher = they avoid each other)", fontsize=14, fontweight='bold')
    plt.ylabel("Average distance to neighbors (Pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "Macro_Scattering.png"), dpi=150)
    plt.close()

    # 2. Number of Tribes Graph
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=super_df, x="Iteration", y="Num_Tribes", hue="Condition", linewidth=2, errorbar=None)
    plt.title("Formation of Distinct Tribes (DBSCAN Clusters)", fontsize=14, fontweight='bold')
    plt.ylabel("Average number of dense groups")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "Macro_Num_Tribes.png"), dpi=150)
    plt.close()

    # 3. Percentage of Isolated (Noise)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=super_df, x="Iteration", y="Isolated_Percentage", hue="Condition", linewidth=2, errorbar=None)
    plt.title("Rate of Isolated Individuals (Spatial wandering)", fontsize=14, fontweight='bold')
    plt.ylabel("% of non-tribe agents")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "Macro_Isolated.png"), dpi=150)
    plt.close()
    
    print(f"Comparison generated in folder: {save_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spatial cluster analysis on simulation results.")
    parser.add_argument("--path", type=str, default="results", help="Path to the results folder containing condition runs.")
    args = parser.parse_args()
    
    macro_spatial_df = spatial_analysis(args.path)
    
    if not macro_spatial_df.empty:
        # Save Excel/CSV of data for archives
        macro_spatial_df.to_csv(os.path.join(args.path, "Multi_Runs_Spatial_Base.csv"), index=False)
        draw_spatial_graph(macro_spatial_df, args.path)
    else:
        print("No data to analyze.")
