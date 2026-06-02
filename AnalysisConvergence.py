import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def convergence_speed(parent_folder: str) -> pd.DataFrame:
    conditions = ["Bounded_Condition", "Unbounded_Condition"]
    target_columns = [
        "Eye_Forward", "Eye_Rotate", 
        "Forward_Rotate", "Rotate_Forward", 
        "Forward_Eye", "Rotate_Eye"
    ]
    
    global_data = []
    
    for condition in conditions:
        condition_path = os.path.join(parent_folder, condition)
        run_folders = glob.glob(os.path.join(condition_path, "Run_*"))
        
        for run_folder in run_folders:
            run_name = os.path.basename(run_folder)
            csv_files = glob.glob(os.path.join(run_folder, "exports", "*.csv"))
            
            for file in csv_files:
                file_name = os.path.basename(file)
                try:
                    iteration = int(file_name.split('_')[-1].replace('.csv', ''))
                except ValueError: 
                    continue
                
                df = pd.read_csv(file)
                
                # Convert weights to -1/0/1 (to have the same order of magnitude)
                phenotype_df = np.sign(df[target_columns])
                
                # Calculate the diversity of EACH trait (standard deviation)
                # Then take the mean of these standard deviations
                global_instability = phenotype_df.std().mean()
                
                global_data.append({
                    "Condition": condition,
                    "Seed": run_name,
                    "Iteration": iteration,
                    "Global_Instability": global_instability
                })
                
    # Create Super-Table
    super_df = pd.DataFrame(global_data)
    if not super_df.empty:
        super_df = super_df.sort_values(by="Iteration")
    
    return super_df

def draw_convergence_evolution(super_df: pd.DataFrame, parent_folder: str):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(
        data=super_df, 
        x="Iteration", 
        y="Global_Instability", 
        hue="Condition", 
        linewidth=2.5,
        errorbar=None
    )
    
    plt.title("The Evolutionary Clock: Convergence Speed", fontsize=15, fontweight='bold')
    plt.xlabel("Simulation Iterations")
    plt.ylabel("Global Behavioral Instability")
    
    # Visual reference line for "dead calm"
    plt.axhline(0.1, color='black', linestyle='--', alpha=0.5, label="Stability Threshold (Convergence)")
    plt.legend()
    plt.tight_layout()
    
    save_folder = os.path.join(parent_folder, "Convergence_Analysis")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "Convergence_Speed.png")
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"File generated: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run convergence analysis on simulation results.")
    parser.add_argument("--path", type=str, default="results", help="Path to the results folder containing condition runs.")
    args = parser.parse_args()
    
    convergence_df = convergence_speed(args.path)
    if not convergence_df.empty:
        draw_convergence_evolution(convergence_df, args.path)
    else:
        print("No data to analyze.")
