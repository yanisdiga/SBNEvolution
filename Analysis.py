import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
import itertools
import argparse

# ==========================================
# 1. GRAPH GENERATORS
# ==========================================

def generate_temporal_dynamics(df_total: pd.DataFrame, save_folder: str):
    """Generates and saves the evolution graph over the entire simulation."""
    phenotype_columns = [
        "Eye_Forward", "Eye_Rotate",
        "Forward_Rotate", "Rotate_Forward",
        "Forward_Eye", "Rotate_Eye"
    ]
    
    df_melted = df_total.melt(
        id_vars=['Iteration'], 
        value_vars=phenotype_columns,
        var_name='Connection', 
        value_name='Weight'
    )

    # Global graph (displaying everything)
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_melted, x='Iteration', y='Weight', hue='Connection', linewidth=2, errorbar=None)

    plt.title("Global Evolutionary Dynamics of Neural Networks", fontsize=16, fontweight='bold')
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Average influence weight", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    
    global_path = os.path.join(save_folder, "global_evolution.png")
    plt.savefig(global_path, dpi=150)
    plt.close()

    # Comparison graphs (Pairwise)
    pairs = list(itertools.combinations(phenotype_columns, 2))
    total_pairs = len(pairs)
    print(f"  Generating {len(pairs)} comparison graphs...")
    for i, (trait_A, trait_B) in enumerate(pairs, start=1):
        # Filter data to keep only the two targeted traits
        df_pair = df_melted[df_melted['Connection'].isin([trait_A, trait_B])]
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_pair, x='Iteration', y='Weight', hue='Connection', linewidth=2, palette=['#1f77b4', '#ff7f0e'], errorbar=None)
        
        plt.title(f"Evolutionary Competition: {trait_A} vs {trait_B}", fontsize=14, fontweight='bold')
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Average influence weight", fontsize=12)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        
        # Save
        filename = f"comparison_{trait_A}_vs_{trait_B}.png"
        pair_path = os.path.join(save_folder, filename)
        plt.savefig(pair_path, dpi=150)
        plt.close()
        
        # Display progress
        print(f"[{i}/{total_pairs}] Comparison {trait_A} vs {trait_B} completed.")


def generate_spatial_maps(df: pd.DataFrame, iteration: int, save_path: str, width: int = 1280, height: int = 720):
    """Generates and saves the spatial mosaic for a specific epoch."""
    phenotype_traits = [
        "Eye_Forward", "Eye_Rotate",
        "Forward_Rotate", "Rotate_Forward",
        "Forward_Eye", "Rotate_Eye"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    fig.suptitle(f"Ecological Maps (Iteration {iteration})\n{len(df)} agents", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for i, trait in enumerate(phenotype_traits):
        ax = axes[i]
        if trait not in df.columns:
            continue
            
        scatter = ax.scatter(
            x=df['Pos_X'], y=df['Pos_Y'], c=df[trait], 
            cmap='coolwarm', s=40, alpha=0.8, edgecolors='black', linewidth=0.5
        )
        
        ax.set_title(f"Trait: {trait}", fontsize=14, fontweight='bold')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_averages_barplot(df: pd.DataFrame, iteration: int, save_path: str):
    """Generates and saves the averages barplot for a specific epoch."""
    phenotype_columns = [
        "Eye_Forward", "Eye_Rotate",
        "Forward_Rotate", "Rotate_Forward",
        "Forward_Eye", "Rotate_Eye"
    ]
    
    df_melted = df.melt(value_vars=phenotype_columns, var_name='Connection', value_name='Weight')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Connection', y='Weight', hue='Connection', palette='magma', legend=False, capsize=0.1)

    plt.title(f"Average Synaptic Influences (Iteration {iteration})", fontsize=15, fontweight='bold')
    plt.xlabel("Connection Pair", fontsize=12)
    plt.ylabel("Average influence weight", fontsize=12)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2) 
    plt.xticks(rotation=15, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()

# ==========================================
# 2. PIPELINE
# ==========================================

def launch_analysis_pipeline(simulation_folder: str):
    print("=" * 50)
    print(f"LAUNCHING ANALYSIS PIPELINE")
    print(f"Target: {simulation_folder}")
    print("=" * 50)

    # Definition and creation of the tree structure
    exports_folder = os.path.join(simulation_folder, "exports")
    analysis_folder = os.path.join(simulation_folder, "analysis")
    
    temporal_dir = os.path.join(analysis_folder, "1_temporal_dynamics")
    spatial_dir = os.path.join(analysis_folder, "2_spatial_maps")
    averages_dir = os.path.join(analysis_folder, "3_averages_barplot")
    
    for folder in [temporal_dir, spatial_dir, averages_dir]:
        os.makedirs(folder, exist_ok=True)

    # Fetch files
    files = glob.glob(os.path.join(exports_folder, "agents_save_*.csv"))
    if not files:
        print(f"No CSV file found in {exports_folder}")
        return
        
    print(f"{len(files)} files detected. Starting processing...")

    global_df_list = []

    # File by file processing loop
    for index, file in enumerate(files):
        match = re.search(r'agents_save_(\d+)\.csv', file)
        if match:
            iteration = int(match.group(1))
            
            # Load DataFrame
            current_df = pd.read_csv(file)
            current_df['Iteration'] = iteration
            global_df_list.append(current_df)
            
            # Save paths
            spatial_path = os.path.join(spatial_dir, f"spatial_{iteration:08d}.png")
            average_path = os.path.join(averages_dir, f"averages_{iteration:08d}.png")
            
            # Individual graphs generation
            generate_spatial_maps(current_df, iteration, spatial_path)
            generate_averages_barplot(current_df, iteration, average_path)
            
            print(f"  [{index+1}/{len(files)}] Iteration {iteration} processed.")

    # 4. Global temporal processing
    print("\nMerging data for temporal analysis...")
    df_total = pd.concat(global_df_list, ignore_index=True)
    df_total.sort_values(by='Iteration', inplace=True)
    
    temporal_path = os.path.join(temporal_dir, "global_evolution.png")
    generate_temporal_dynamics(df_total, temporal_dir)
    print("  Global evolution generated.")

    print("\n" + "=" * 50)
    print("ANALYSIS SUCCESSFULLY COMPLETED")
    print(f"All graphs are saved in: {analysis_folder}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation data.")
    parser.add_argument("--path", type=str, required=True, help="Path to the simulation results folder (e.g. results/Test_1)")
    args = parser.parse_args()
    
    sns.set_theme(style="whitegrid")
    launch_analysis_pipeline(args.path)
