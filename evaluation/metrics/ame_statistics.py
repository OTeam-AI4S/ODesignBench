"""
AME (Atomic Motif Enzyme) evaluation statistics and visualization.

This module provides functions to:
1. Parse AME evaluation results CSV
2. Aggregate results by design and target
3. Generate statistics and visualizations
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings


def parse_sample_name(sample_name: str) -> Tuple[str, str]:
    """
    Parse sample name to extract target and design key.
    
    Args:
        sample_name: Sample name like "m0129_1os7_seed_46_bb_2_seq_0-1-6"
        
    Returns:
        Tuple of (target, design_key)
        - target: e.g., "1os7" (from split by '_' and take index 1)
        - design_key: e.g., "m0129_1os7_seed_46_bb_2_seq_0-1" (remove last '-{1-8}' suffix)
    """
    parts = str(sample_name).split('_')
    if len(parts) < 2:
        return 'Unknown', sample_name
    
    # Extract target (second part after splitting by '_')
    target = parts[1]
    
    # Extract design key by removing the last '-{1-8}' suffix
    # Example: "m0129_1os7_seed_46_bb_2_seq_0-1-6" -> "m0129_1os7_seed_46_bb_2_seq_0-1"
    design_key = str(sample_name).rsplit('-', 1)[0]
    
    return target, design_key


def aggregate_ame_results(
    csv_path: str,
    success_col: str = 'success'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate AME evaluation results by design and target.
    
    Args:
        csv_path: Path to AME evaluation results CSV
        success_col: Name of the success column (default: 'success')
        
    Returns:
        Tuple of (design_stats, target_stats) DataFrames
        - design_stats: Success status for each design (one row per design)
        - target_stats: Aggregated statistics per target
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Handle first column (index)
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    elif df.columns[0] not in df.columns[1:]:
        # First column is likely the index
        df.rename(columns={df.columns[0]: 'id'}, inplace=True)
    
    # If no 'id' column, use index
    if 'id' not in df.columns:
        df.reset_index(inplace=True)
        if 'index' in df.columns:
            df.rename(columns={'index': 'id'}, inplace=True)
        else:
            df['id'] = df.index
    
    # Parse target and design key
    df[['Target', 'Design_Key']] = df['id'].apply(
        lambda x: pd.Series(parse_sample_name(x))
    )
    
    # Ensure success column is boolean
    if success_col not in df.columns:
        # Try to compute success from metrics if available
        if 'catalytic_heavy_atom_rmsd' in df.columns and 'ligand_clash_count_1_5A' in df.columns:
            catalytic_pass = df['catalytic_heavy_atom_rmsd'].notna() & (df['catalytic_heavy_atom_rmsd'] < 1.5)
            clash_pass = df['ligand_clash_count_1_5A'] == 0
            df[success_col] = catalytic_pass & clash_pass
        else:
            warnings.warn(f"Success column '{success_col}' not found and cannot be computed. Using False for all.")
            df[success_col] = False
    
    df[success_col] = df[success_col].astype(bool)
    
    # Aggregate to design level: at least 1 sequence out of 8 is successful
    design_stats = df.groupby(['Target', 'Design_Key'])[success_col].any().reset_index()
    design_stats.rename(columns={success_col: 'Design_Success'}, inplace=True)
    
    # Aggregate to target level
    target_stats = design_stats.groupby('Target')['Design_Success'].agg(['sum', 'count']).reset_index()
    target_stats.rename(columns={'sum': 'Success_Count', 'count': 'Total_Designs'}, inplace=True)
    target_stats['Success_Rate'] = (target_stats['Success_Count'] / target_stats['Total_Designs'] * 100)
    
    # Sort by success rate descending
    target_stats = target_stats.sort_values('Success_Rate', ascending=False)
    
    return design_stats, target_stats


def visualize_ame_results(
    target_stats: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7),
    dpi: int = 150,
    title: Optional[str] = None
) -> None:
    """
    Generate visualization of AME evaluation results by target.
    
    Args:
        target_stats: DataFrame with target-level statistics (from aggregate_ame_results)
        output_path: Path to save the figure (if None, uses 'ame_target_success_rates.png')
        figsize: Figure size (width, height)
        dpi: DPI for saved figure
        title: Custom title for the plot (if None, uses default)
    """
    # Set style
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['figure.dpi'] = dpi
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create bar plot
    ax = sns.barplot(
        data=target_stats,
        x='Target',
        y='Success_Rate',
        palette='viridis'
    )
    
    # Annotate bars with count/total and percentage
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if i < len(target_stats):
            row = target_stats.iloc[i]
            count = int(row['Success_Count'])
            total = int(row['Total_Designs'])
            rate = row['Success_Rate']
            
            # Annotate above the bar
            ax.annotate(
                f'{count}/{total}\n{rate:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                xytext=(0, 3),
                textcoords='offset points'
            )
    
    # Set title
    if title is None:
        title = (
            "Design-Level Success Rate by Target (AME Evaluation)\n"
            "Criteria: RMSD < 1.5 Å and 0 Clashes (At least 1 success per 8 sequences)"
        )
    plt.title(title)
    
    # Set labels
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Target")
    plt.xticks(rotation=45)
    
    # Expand y-limit slightly to fit annotations
    max_rate = target_stats['Success_Rate'].max()
    plt.ylim(0, max_rate + (max_rate * 0.15) if max_rate > 0 else 10)
    
    # Save figure
    plt.tight_layout()
    if output_path is None:
        output_path = 'ame_target_success_rates.png'
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def generate_ame_statistics(
    csv_path: str,
    output_dir: Optional[str] = None,
    success_col: str = 'success',
    save_plots: bool = True,
    save_stats: bool = True
) -> Dict:
    """
    Generate comprehensive AME evaluation statistics and visualizations.
    
    Args:
        csv_path: Path to AME evaluation results CSV
        output_dir: Directory to save outputs (if None, uses CSV directory)
        success_col: Name of the success column
        save_plots: Whether to save visualization plots
        save_stats: Whether to save statistics CSV files
        
    Returns:
        Dictionary containing statistics summary
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results
    print(f"Loading AME evaluation results from {csv_path}...")
    design_stats, target_stats = aggregate_ame_results(csv_path, success_col=success_col)
    
    # Print statistics
    print("\n" + "="*60)
    print("AME Evaluation Statistics")
    print("="*60)
    print("\n--- Target-Level Success Statistics ---")
    print(target_stats.to_string(index=False))
    
    # Overall statistics
    total_designs = target_stats['Total_Designs'].sum()
    total_success = target_stats['Success_Count'].sum()
    overall_rate = (total_success / total_designs * 100) if total_designs > 0 else 0
    
    print(f"\n--- Overall Statistics ---")
    print(f"Total Designs: {total_designs}")
    print(f"Successful Designs: {total_success}")
    print(f"Overall Success Rate: {overall_rate:.2f}%")
    print(f"Number of Targets: {len(target_stats)}")
    
    # Save statistics CSV files
    if save_stats:
        design_stats_path = output_dir / 'ame_design_statistics.csv'
        target_stats_path = output_dir / 'ame_target_statistics.csv'
        
        design_stats.to_csv(design_stats_path, index=False)
        target_stats.to_csv(target_stats_path, index=False)
        
        print(f"\nStatistics saved:")
        print(f"  - Design-level: {design_stats_path}")
        print(f"  - Target-level: {target_stats_path}")
    
    # Generate visualization
    if save_plots:
        plot_path = output_dir / 'ame_target_success_rates.png'
        visualize_ame_results(target_stats, output_path=str(plot_path))
    
    # Return summary dictionary
    summary = {
        'total_designs': int(total_designs),
        'total_success': int(total_success),
        'overall_success_rate': float(overall_rate),
        'num_targets': len(target_stats),
        'target_statistics': target_stats.to_dict('records')
    }
    
    return summary
