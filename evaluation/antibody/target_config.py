"""
Target Configuration Loader

Loads target information from target_config.csv and provides utilities for
determining Part 1 vs Part 2 targets and scaffold requirements.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


def load_target_config(config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load target configuration from CSV file.
    
    Args:
        config_path: Path to target_config.csv. If None, uses default path.
        
    Returns:
        DataFrame with target configuration
    """
    if config_path is None:
        # Default path relative to this file
        default_path = Path(__file__).parent.parent.parent.parent / "assets" / "antibody_nanobody" / "config" / "target_config.csv"
        config_path = str(default_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Target config file not found: {config_path}")
    
    df = pd.read_csv(config_path)
    
    # Validate required columns
    required_cols = ['target_id', 'antigen_chains', 'target_hotspots', 'epitope_description']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in target config: {missing_cols}")
    
    return df


def get_part1_targets(df: pd.DataFrame) -> List[str]:
    """
    Get Part 1 target IDs (targets with binding hotspots: 1-11, 21, 22).
    
    Args:
        df: Target configuration DataFrame
        
    Returns:
        List of target IDs (e.g., ['01_7UXQ', '02_1TNF', ...])
    """
    # Part 1: targets with non-empty target_hotspots (1-11, 21, 22)
    part1_mask = df['target_hotspots'].notna() & (df['target_hotspots'] != '')
    part1_targets = df[part1_mask]['target_id'].tolist()
    return sorted(part1_targets)


def get_part2_targets(df: pd.DataFrame) -> List[str]:
    """
    Get Part 2 target IDs (targets without binding hotspots: 12-20).
    
    Args:
        df: Target configuration DataFrame
        
    Returns:
        List of target IDs (e.g., ['12_1BI7', '13_1G1D', ...])
    """
    # Part 2: targets with empty target_hotspots (12-20)
    part2_mask = df['target_hotspots'].isna() | (df['target_hotspots'] == '')
    part2_targets = df[part2_mask]['target_id'].tolist()
    return sorted(part2_targets)


def get_target_info(df: pd.DataFrame, target_id: str) -> Optional[Dict]:
    """
    Get information for a specific target.
    
    Args:
        df: Target configuration DataFrame
        target_id: Target ID (e.g., '01_7UXQ')
        
    Returns:
        Dictionary with target information or None if not found
    """
    matches = df[df['target_id'] == target_id]
    if len(matches) == 0:
        return None
    
    row = matches.iloc[0]
    return {
        'target_id': row['target_id'],
        'antigen_chains': row['antigen_chains'],
        'target_hotspots': row['target_hotspots'] if pd.notna(row['target_hotspots']) else None,
        'epitope_description': row['epitope_description']
    }


def is_part1_target(df: pd.DataFrame, target_id: str) -> bool:
    """
    Check if a target is Part 1 (has binding hotspots).
    
    Args:
        df: Target configuration DataFrame
        target_id: Target ID to check
        
    Returns:
        True if Part 1, False if Part 2
    """
    part1_targets = get_part1_targets(df)
    return target_id in part1_targets
