import argparse
import os
import sys
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging
import traceback
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from collections import Counter
from glob import glob
import json
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)

# Import required modules
from rdkit import Chem

import Bio.PDB
from Bio.PDB import MMCIFIO, PDBIO, Select
import scipy.spatial
import string

from rdkit import rdBase

logger = logging.getLogger(__name__)
rdkit_logger = logging.getLogger('rdkit')

from metrics.ligand.geometry import eval_torsion_angle, eval_bond_length, eval_stability, eval_bond_angle, eval_steric_clash
from metrics.ligand.geometry.eval_bond_length_config import set_ccd_bond_length_path
from metrics.ligand.geometry.eval_bond_angle_config import set_ccd_bond_angle_path
from metrics.ligand.geometry.eval_torsion_angle_config import set_ccd_torsion_angle_path
from metrics.ligand import scoring
from metrics.ligand.datasets.parsers import process_single_cif

# ===== Evaluation functions =====
def evaluate_geometry(ligand_file, pocket_file, output_dir):
    """Evaluate geometry properties of a ligand"""
    try:
        rdkit_logger.info(f'Evaluate_geometry Chem.SDMolSupplier {ligand_file} to rdkit mol...')
        mol = Chem.SDMolSupplier(ligand_file)[0]
        if mol is None:
            logger.warning(f"warning: cannot read molecule file {ligand_file}")
            return None
        
        # Check basic molecule information
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        logger.debug(f"molecule information: atom number={num_atoms}, bond number={num_bonds}")
        
        if num_atoms == 0:
            logger.warning(f"warning: molecule {ligand_file} has no atoms")
            return None
        
        atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        pos = mol.GetConformer().GetPositions()
        
        # Check if atom coordinates are valid
        if pos is None or len(pos) == 0:
            logger.warning(f"warning: molecule {ligand_file} has no valid atom coordinates")
            return None
        
        # Evaluate bond length with detailed error handling
        try:
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            logger.debug(f"bond length evaluation successful, obtained {len(bond_dist) if bond_dist else 0} bond lengths")
        except Exception as e:
            logger.error(f"bond length evaluation failed: {e}")
            logger.error(f"molecule file: {ligand_file}")
            logger.error(f"molecule atom number: {num_atoms}, bond number: {num_bonds}")
            # Try to get more detailed error information
            try:
                # Check bond information in the molecule
                bonds = mol.GetBonds()
                for i, bond in enumerate(bonds):
                    begin_atom = bond.GetBeginAtom()
                    end_atom = bond.GetEndAtom()
                    logger.debug(f"bond {i}: {begin_atom.GetSymbol()}({begin_atom.GetIdx()}) - {end_atom.GetSymbol()}({end_atom.GetIdx()})")
            except Exception as bond_e:
                logger.error(f"cannot check bond information: {bond_e}")
            bond_dist = []
        
        # Evaluate bond angle with detailed error handling
        try:
            bond_angle = eval_bond_angle.bond_angle_from_mol(mol)
            logger.debug(f"bond angle evaluation successful, obtained {len(bond_angle) if bond_angle else 0} bond angles")
        except Exception as e:
            logger.error(f"bond angle evaluation failed: {ligand_file} {e}")
            bond_angle = []
        
        # Evaluate torsion angle with detailed error handling
        try:
            
            torsion_angle = eval_torsion_angle.torsion_angle_from_mol(mol)
            logger.debug(f"torsion angle evaluation successful, obtained {len(torsion_angle) if torsion_angle else 0} torsion angles")
        except Exception as e:
            logger.error(f"torsion angle evaluation failed: {ligand_file} {e}")
            torsion_angle = []
        
        # breakpoint()
        # Evaluate stability with detailed error handling
        try:
            r_stable = eval_stability.check_stability(pos, atom_types)
            logger.debug(f"stability evaluation successful: {r_stable}")
        except Exception as e:
            logger.error(f"stability evaluation failed: {ligand_file} {e}")
            r_stable = [False, 0, num_atoms]  # Default value
        
        # Evaluate clash with detailed error handling
        try:
            clash_detected, clash_info = eval_steric_clash.eval_steric_clash(mol, pdb_file=pocket_file)
            logger.debug(f"clash detection successful: {clash_detected}")
        except Exception as e:
            logger.error(f"clash detection failed: {ligand_file} {e}")
            # Provide default clash information
            clash_info = {
                'lig_pro_clash_detected': False,
                'lig_lig_clash': {'clash_atom_num': 0, 'clashed_indices': [], 'clashed_distances': [], 'clashed_vdw_sums': []},
                'lig_pro_clash': {'clash_atom_num': 0, 'clashed_indices': [], 'clashed_distances': [], 'clashed_vdw_sums': []}
            }
            clash_detected = False
        
        result = {
            'ligand_file': ligand_file,
            'pocket_file': pocket_file,
            'bond_dist': bond_dist,
            'bond_angle': bond_angle,
            'torsion_angle': torsion_angle,
            'mol_stable': r_stable[0] if len(r_stable) > 0 else False,
            'atom_stable_num': r_stable[1] if len(r_stable) > 1 else 0,
            'atom_num': r_stable[2] if len(r_stable) > 2 else num_atoms,
            'clash_detected': clash_info.get('lig_pro_clash_detected', False),
            'intra_clash_atom_num': clash_info.get('lig_lig_clash', {}).get('clash_atom_num', 0),
            'inter_clash_atom_num': clash_info.get('lig_pro_clash', {}).get('clash_atom_num', 0),
            # Add detailed protein-ligand clash information
            'lig_pro_clash_indices': clash_info.get('lig_pro_clash', {}).get('clashed_indices', []),
            'lig_pro_clash_distances': clash_info.get('lig_pro_clash', {}).get('clashed_distances', []),
            'lig_pro_clash_vdw_sums': clash_info.get('lig_pro_clash', {}).get('clashed_vdw_sums', []),
            # Add detailed ligand intra-molecular clash information
            'lig_lig_clash_indices': clash_info.get('lig_lig_clash', {}).get('clashed_indices', []),
            'lig_lig_clash_distances': clash_info.get('lig_lig_clash', {}).get('clashed_distances', []),
            'lig_lig_clash_vdw_sums': clash_info.get('lig_lig_clash', {}).get('clashed_vdw_sums', [])
        }
        
        return result
        
    except Exception as e:
        logger.error(f"error: geometry evaluation failed {ligand_file}: {e}")
        logger.error(f"detailed error information: {traceback.format_exc()}")
        return None


def evaluate_chemistry(ligand_file, pocket_file, output_dir, center=None, exhaustiveness=16, enable_vina=True):
    """Evaluate chemistry properties of a ligand"""
    try:
        rdkit_logger.info(f'Evaluate_chemistry Chem.SDMolSupplier {ligand_file} to rdkit mol...')
        mol = Chem.SDMolSupplier(ligand_file)[0]
        if mol is None:
            return None
        
        smiles = Chem.MolToSmiles(mol)
        chem_results = scoring.get_chem(mol)
        
        result = {
            'mol': mol,
            'smiles': smiles,
            'ligand_file': ligand_file,
            'pocket_file': pocket_file,
            'chem_results': chem_results,
            'num_atoms': mol.GetNumAtoms()
        }
        
        # Docking evaluation (optional based on enable_vina flag)
        if enable_vina:
            try:
                from metrics.ligand.docking_vina import VinaDockingTask

                dock_save_dir = os.path.join(output_dir, 'docking_results')
                os.makedirs(dock_save_dir, exist_ok=True)
                
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, ligand_file, protein_path=pocket_file, center=center)
                
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness, save_dir=dock_save_dir)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness, save_dir=dock_save_dir)
                docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness,save_dir=dock_save_dir)
                
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results,
                    'dock': docking_results
                }
                result['vina'] = vina_results
            except ModuleNotFoundError as e:
                logger.warning(f"Vina dependencies unavailable, skipping docking for {ligand_file}: {e}")
                result['vina'] = None
        else:
            result['vina'] = None
        
        return result
        
    except Exception as e:
        logger.error(f"Error: Chemistry evaluation failed {ligand_file}: {e}")
        return None


def process_single_ligand_pocket_pair(args_tuple):
    """Process a single ligand-pocket pair for evaluation"""
    ligand_file, pocket_file, output_dir, center, exhaustiveness, enable_geom, enable_chem, enable_vina = args_tuple
    
    try:
        geom_result = None
        chem_result = None
        
        # Geometry evaluation (optional)
        if enable_geom:
            geom_result = evaluate_geometry(ligand_file, pocket_file, output_dir)
        
        # Chemistry evaluation (optional)
        if enable_chem:
            chem_result = evaluate_chemistry(ligand_file, pocket_file, output_dir, center, exhaustiveness, enable_vina)
        
        return {
            'ligand_file': ligand_file,
            'pocket_file': pocket_file,
            'geom_result': geom_result,
            'chem_result': chem_result
        }
        
    except Exception as e:
        logger.error(f"error: process ligand-pocket pair failed {ligand_file}: {e}")
        return None


def collect_and_save_results(results, output_dir, enable_geom=True, enable_chem=True, enable_vina=True):
    """Collect all results and save to files"""
    # Filter out None results - now more flexible about which components are required
    valid_results = []
    failed_geom_count = 0
    failed_chem_count = 0
    
    for r in results:
        if r is not None:
            # Check if at least one evaluation was successful
            has_valid_geom = not enable_geom or r['geom_result'] is not None
            has_valid_chem = not enable_chem or r['chem_result'] is not None
            
            if not has_valid_geom and enable_geom:
                failed_geom_count += 1
            if not has_valid_chem and enable_chem:
                failed_chem_count += 1
            
            if has_valid_geom and has_valid_chem:
                valid_results.append(r)
    
    if not valid_results:
        logger.warning("warning: no valid results")
        return
    
    logger.info(f"collected {len(valid_results)} valid results")
    if failed_geom_count > 0:
        logger.warning(f"geometry evaluation failed: {failed_geom_count} samples")
    if failed_chem_count > 0:
        logger.warning(f"chemistry evaluation failed: {failed_chem_count} samples")
    
    summary = {
        'total_samples': len(valid_results),
        'failed_geometry_count': failed_geom_count,
        'failed_chemistry_count': failed_chem_count,
        'evaluation_modules': {
            'geometry_enabled': enable_geom,
            'chemistry_enabled': enable_chem,
            'vina_enabled': enable_vina
        }
    }
    
    # Geometry statistics (if enabled)
    if enable_geom:
        # count all geometry evaluation attempts (including failed ones)
        all_geom_attempts = []
        for r in results:
            if r is not None and enable_geom:
                all_geom_attempts.append(r)
        
        geom_results = [r['geom_result'] for r in all_geom_attempts if r['geom_result'] is not None]
        failed_geom_results = [r for r in all_geom_attempts if r['geom_result'] is None]
        
        if all_geom_attempts:
            all_bond_dist = []
            all_bond_angle = []
            all_torsion_angle = []
            all_mol_stable = 0
            all_atom_stable = 0
            all_n_atom = 0
            n_clash_mol = 0
            n_quality_pass = 0  # Count of samples that pass quality checks (stable and no clash)
            intra_clash_atom_num = []
            inter_clash_atom_num = []
            
            for geom in geom_results:
                # bond length evaluation
                if geom.get('bond_dist') and len(geom['bond_dist']) > 0:
                    all_bond_dist.extend(geom['bond_dist'])
                
                # bond angle evaluation
                if geom.get('bond_angle') and len(geom['bond_angle']) > 0:
                    all_bond_angle.extend(geom['bond_angle'])
                
                # torsion angle evaluation
                if geom.get('torsion_angle') and len(geom['torsion_angle']) > 0:
                    all_torsion_angle.extend(geom['torsion_angle'])
                
                # stability evaluation
                is_mol_stable = False
                if 'mol_stable' in geom and geom['mol_stable'] is not None:
                    is_mol_stable = bool(geom['mol_stable'])
                    if is_mol_stable:
                        all_mol_stable += 1
                
                # clash detection
                has_clash = False
                if 'clash_detected' in geom and geom['clash_detected'] is not None:
                    has_clash = bool(geom['clash_detected'])
                    if has_clash:
                        n_clash_mol += 1
                
                # Quality check: stable and no clash
                if is_mol_stable and not has_clash:
                    n_quality_pass += 1
                
                # atom statistics
                if 'atom_stable_num' in geom and geom['atom_stable_num'] is not None:
                    all_atom_stable += geom['atom_stable_num']
                if 'atom_num' in geom and geom['atom_num'] is not None:
                    all_n_atom += geom['atom_num']
                
                # clash atom number
                if 'intra_clash_atom_num' in geom and geom['intra_clash_atom_num'] is not None:
                    intra_clash_atom_num.append(geom['intra_clash_atom_num'])
                if 'inter_clash_atom_num' in geom and geom['inter_clash_atom_num'] is not None:
                    inter_clash_atom_num.append(geom['inter_clash_atom_num'])
            
            # calculate overall success rate
            total_geom_attempts = len(all_geom_attempts)
            total_geom_success = len(geom_results)
            
            # Evaluation success rate: percentage of samples that were successfully evaluated (no errors)
            evaluation_success_rate = total_geom_success / total_geom_attempts if total_geom_attempts > 0 else 0
            
            # Quality success rate: percentage of successfully evaluated samples that pass quality checks (stable and no clash)
            quality_success_rate = n_quality_pass / total_geom_success if total_geom_success > 0 else 0
            
            summary['geometry'] = {
                'total_attempts': total_geom_attempts,
                'total_success': total_geom_success,
                'overall_success_rate': evaluation_success_rate,  # Renamed for clarity
                'evaluation_success_rate': evaluation_success_rate,  # Percentage of samples successfully evaluated
                'quality_success_rate': quality_success_rate,  # Percentage of evaluated samples that pass quality checks
                'quality_pass_count': n_quality_pass  # Number of samples that are stable and have no clash
            }
            
            # calculate statistics only when there are data
            if all_bond_dist:
                summary['geometry']['fraction_mol_stable'] = all_mol_stable / total_geom_success if total_geom_success > 0 else 0
                summary['geometry']['fraction_atom_stable'] = all_atom_stable / all_n_atom if all_n_atom > 0 else 0
                summary['geometry']['intra_clash_atom_ratio'] = np.sum(intra_clash_atom_num) / all_n_atom if all_n_atom > 0 else 0
                summary['geometry']['inter_clash_atom_ratio'] = np.sum(inter_clash_atom_num) / all_n_atom if all_n_atom > 0 else 0
                summary['geometry']['clash_mol_ratio'] = n_clash_mol / total_geom_success if total_geom_success > 0 else 0
            else:
                summary['geometry']['fraction_mol_stable'] = 0
                summary['geometry']['fraction_atom_stable'] = 0
                summary['geometry']['intra_clash_atom_ratio'] = 0
                summary['geometry']['inter_clash_atom_ratio'] = 0
                summary['geometry']['clash_mol_ratio'] = 0
            
            # Calculate overall JSD for bond length, bond angle, and torsion angle
            overall_jsd_metrics = {}
            
            # Bond length overall JSD
            if all_bond_dist:
                try:
                    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
                    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
                    # Calculate overall JSD for bond length
                    valid_jsd_values = [v for v in c_bond_length_dict.values() if v is not None]
                    if valid_jsd_values:
                        overall_jsd_metrics['overall_bond_length_jsd'] = np.mean(valid_jsd_values)
                        overall_jsd_metrics['overall_bond_length_jsd_std'] = np.std(valid_jsd_values)
                    else:
                        overall_jsd_metrics['overall_bond_length_jsd'] = None
                        overall_jsd_metrics['overall_bond_length_jsd_std'] = None
                except Exception as e:
                    logger.warning(f"bond length JSD calculation failed: {e}")
                    overall_jsd_metrics['overall_bond_length_jsd'] = None
                    overall_jsd_metrics['overall_bond_length_jsd_std'] = None
            
            # Bond angle overall JSD
            if all_bond_angle:
                try:         
                    c_bond_angle_profile = eval_bond_angle.get_bond_angle_profile(all_bond_angle)
                    c_bond_angle_dict = eval_bond_angle.eval_bond_angle_profile(c_bond_angle_profile)
                    # Calculate overall JSD for bond angle
                    valid_jsd_values = [v for v in c_bond_angle_dict.values() if v is not None]
                    if valid_jsd_values:
                        overall_jsd_metrics['overall_bond_angle_jsd'] = np.mean(valid_jsd_values)
                        overall_jsd_metrics['overall_bond_angle_jsd_std'] = np.std(valid_jsd_values)
                    else:
                        overall_jsd_metrics['overall_bond_angle_jsd'] = None
                        overall_jsd_metrics['overall_bond_angle_jsd_std'] = None
                except Exception as e:
                    logger.warning(f"bond angle JSD calculation failed: {e}")
                    overall_jsd_metrics['overall_bond_angle_jsd'] = None
                    overall_jsd_metrics['overall_bond_angle_jsd_std'] = None
            
            # Torsion angle overall JSD
            if all_torsion_angle:
                try:

                    c_torsion_angle_profile = eval_torsion_angle.get_torsion_angle_profile(all_torsion_angle)
                    c_torsion_angle_dict = eval_torsion_angle.eval_torsion_angle_profile(c_torsion_angle_profile)
                    # Calculate overall JSD for torsion angle
                    valid_jsd_values = [v for v in c_torsion_angle_dict.values() if v is not None]
                    if valid_jsd_values:
                        overall_jsd_metrics['overall_torsion_angle_jsd'] = np.mean(valid_jsd_values)
                        overall_jsd_metrics['overall_torsion_angle_jsd_std'] = np.std(valid_jsd_values)
                    else:
                        overall_jsd_metrics['overall_torsion_angle_jsd'] = None
                        overall_jsd_metrics['overall_torsion_angle_jsd_std'] = None
                except Exception as e:
                    logger.warning(f"torsion angle JSD calculation failed: {e}")
                    overall_jsd_metrics['overall_torsion_angle_jsd'] = None
                    overall_jsd_metrics['overall_torsion_angle_jsd_std'] = None
            
            summary['geometry']['overall_jsd_metrics'] = overall_jsd_metrics
    
    # Chemistry statistics (if enabled)
    if enable_chem:
        chem_results = [r['chem_result'] for r in valid_results if r['chem_result'] is not None]
        if chem_results:
            qed = [r['chem_results']['qed'] for r in chem_results]
            sa = [r['chem_results']['sa'] for r in chem_results]
            
            chemistry_stats = {
                'qed_mean': np.mean(qed),
                'qed_median': np.median(qed),
                'sa_mean': np.mean(sa),
                'sa_median': np.median(sa),
            }
            
            # Vina statistics (if enabled)
            if enable_vina:
                vina_results = [r for r in chem_results if r['vina'] is not None]
                if vina_results:
                    vina_score_only = [r['vina']['score_only']['affinity'] for r in vina_results]
                    vina_min = [r['vina']['minimize']['affinity'] for r in vina_results]
                    vina_dock = [r['vina']['dock']['affinity'] for r in vina_results]
                    
                    chemistry_stats.update({
                        'vina_score_only_mean': np.mean(vina_score_only),
                        'vina_score_only_median': np.median(vina_score_only),
                        'vina_min_mean': np.mean(vina_min),
                        'vina_min_median': np.median(vina_min),
                        'vina_dock_mean': np.mean(vina_dock),
                        'vina_dock_median': np.median(vina_dock)
                    })
            
            summary['chemistry'] = chemistry_stats
    
    # Save detailed results
    # torch.save(valid_results, os.path.join(output_dir, 'evaluation_results.pt'))
    torch.save(summary, os.path.join(output_dir, 'evaluation_summary.pt'))
    
    # Create DataFrame for easy viewing
    df_data = []
    for r in valid_results:
        ligand_name = os.path.basename(r['ligand_file'])
        pocket_name = os.path.basename(r['pocket_file'])
        
        row_data = {
            'ligand_file': ligand_name,
            'pocket_file': pocket_name,
        }
        
        # Add chemistry data if available
        if enable_chem and r['chem_result'] is not None:
            row_data.update({
                'smiles': r['chem_result']['smiles'],
                'qed': r['chem_result']['chem_results']['qed'],
                'sa': r['chem_result']['chem_results']['sa'],
                'logp': r['chem_result']['chem_results']['logp'],
                'lipinski': r['chem_result']['chem_results']['lipinski'],
            })
            
            # Add Vina data if available
            if enable_vina and r['chem_result']['vina'] is not None:
                row_data.update({
                    'vina_dock': r['chem_result']['vina']['dock']['affinity'],
                    'vina_min': r['chem_result']['vina']['minimize']['affinity'],
                    'vina_score_only': r['chem_result']['vina']['score_only']['affinity'],
                })
        
        # Add geometry data if available
        if enable_geom and r['geom_result'] is not None:
            row_data.update({
                'mol_stable': r['geom_result']['mol_stable'],
                'clash_detected': r['geom_result']['clash_detected'],
                'atom_num': r['geom_result']['atom_num'],
                'lig_pro_clash_count': len(r['geom_result']['lig_pro_clash_indices']) if len(r['geom_result']['lig_pro_clash_indices']) > 0 else 0,
                'lig_lig_clash_count': len(r['geom_result']['lig_lig_clash_indices']) if len(r['geom_result']['lig_lig_clash_indices']) > 0 else 0,
                'torsion_angle_count': len(r['geom_result']['torsion_angle']) if 'torsion_angle' in r['geom_result'] else 0
            })
        
        df_data.append(row_data)
    
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    # Print summary
    logger.info("\n=== evaluation summary ===")
    logger.info(f"total samples: {summary['total_samples']}")
    logger.info(f"enabled evaluation modules: geometry={enable_geom}, chemistry={enable_chem}, Vina={enable_vina}")
    
    if enable_geom and 'geometry' in summary:
        logger.info(f"geometry evaluation total attempts: {summary['geometry']['total_attempts']}")
        logger.info(f"geometry evaluation total success: {summary['geometry']['total_success']}")
        logger.info(f"geometry evaluation success rate (evaluation): {summary['geometry']['evaluation_success_rate']:.1%}")
        if 'quality_success_rate' in summary['geometry']:
            logger.info(f"geometry evaluation success rate (quality): {summary['geometry']['quality_success_rate']:.1%} ({summary['geometry'].get('quality_pass_count', 0)}/{summary['geometry']['total_success']} samples pass quality checks)")
        if 'failed_geometry_count' in summary and summary['failed_geometry_count'] > 0:
            logger.warning(f"geometry evaluation failed samples: {summary['failed_geometry_count']}")
        
        logger.info(f"molecule stability: {summary['geometry']['fraction_mol_stable']:.3f}")
        logger.info(f"clash molecule ratio: {summary['geometry']['clash_mol_ratio']:.3f}")
        
        # Print overall JSD metrics
        if 'overall_jsd_metrics' in summary['geometry']:
            jsd_metrics = summary['geometry']['overall_jsd_metrics']
            if jsd_metrics.get('overall_bond_length_jsd') is not None:
                logger.info(f"overall bond length JSD: {jsd_metrics['overall_bond_length_jsd']:.4f} ± {jsd_metrics['overall_bond_length_jsd_std']:.4f}")
            else:
                logger.warning("bond length JSD calculation failed")
            if jsd_metrics.get('overall_bond_angle_jsd') is not None:
                logger.info(f"overall bond angle JSD: {jsd_metrics['overall_bond_angle_jsd']:.4f} ± {jsd_metrics['overall_bond_angle_jsd_std']:.4f}")
            else:
                logger.warning("bond angle JSD calculation failed")
            if jsd_metrics.get('overall_torsion_angle_jsd') is not None:
                logger.info(f"overall torsion angle JSD: {jsd_metrics['overall_torsion_angle_jsd']:.4f} ± {jsd_metrics['overall_torsion_angle_jsd_std']:.4f}")
            else:
                logger.warning("torsion angle JSD calculation failed")
    
    if enable_chem and 'chemistry' in summary:
        logger.info(f"QED: mean={summary['chemistry']['qed_mean']:.3f}, median={summary['chemistry']['qed_median']:.3f}")
        logger.info(f"SA: mean={summary['chemistry']['sa_mean']:.3f}, median={summary['chemistry']['sa_median']:.3f}")
        if enable_vina and 'vina_dock_mean' in summary['chemistry']:
            logger.info(f"Vina Dock: mean={summary['chemistry']['vina_dock_mean']:.3f}, median={summary['chemistry']['vina_dock_median']:.3f}")
    
    # Log summary
    logger.info("=== evaluation summary ===")
    logger.info(f"total samples: {summary['total_samples']}")
    logger.info(f"enabled evaluation modules: geometry={enable_geom}, chemistry={enable_chem}, Vina={enable_vina}")
    
    if enable_geom and 'geometry' in summary:
        logger.info(f"geometry evaluation total attempts: {summary['geometry']['total_attempts']}")
        logger.info(f"geometry evaluation total success: {summary['geometry']['total_success']}")
        logger.info(f"geometry evaluation success rate (evaluation): {summary['geometry']['evaluation_success_rate']:.1%}")
        if 'quality_success_rate' in summary['geometry']:
            logger.info(f"geometry evaluation success rate (quality): {summary['geometry']['quality_success_rate']:.1%} ({summary['geometry'].get('quality_pass_count', 0)}/{summary['geometry']['total_success']} samples pass quality checks)")
        if 'failed_geometry_count' in summary and summary['failed_geometry_count'] > 0:
            logger.warning(f"geometry evaluation failed samples: {summary['failed_geometry_count']}")

        logger.info(f"molecule stability: {summary['geometry']['fraction_mol_stable']:.3f}")
        logger.info(f"clash molecule ratio: {summary['geometry']['clash_mol_ratio']:.3f}")
        
        # Log overall JSD metrics
        if 'overall_jsd_metrics' in summary['geometry']:
            jsd_metrics = summary['geometry']['overall_jsd_metrics']
            if jsd_metrics.get('overall_bond_length_jsd') is not None:
                logger.info(f"overall bond length JSD: {jsd_metrics['overall_bond_length_jsd']:.4f} ± {jsd_metrics['overall_bond_length_jsd_std']:.4f}")
            else:
                logger.warning("bond length JSD calculation failed")
            if jsd_metrics.get('overall_bond_angle_jsd') is not None:
                logger.info(f"overall bond angle JSD: {jsd_metrics['overall_bond_angle_jsd']:.4f} ± {jsd_metrics['overall_bond_angle_jsd_std']:.4f}")
            else:
                logger.warning("bond angle JSD calculation failed")
            if jsd_metrics.get('overall_torsion_angle_jsd') is not None:
                logger.info(f"overall torsion angle JSD: {jsd_metrics['overall_torsion_angle_jsd']:.4f} ± {jsd_metrics['overall_torsion_angle_jsd_std']:.4f}")
            else:
                logger.warning("torsion angle JSD calculation failed")
    
    if enable_chem and 'chemistry' in summary:
        logger.info(f"QED: mean={summary['chemistry']['qed_mean']:.3f}, median={summary['chemistry']['qed_median']:.3f}")
        logger.info(f"SA: mean={summary['chemistry']['sa_mean']:.3f}, median={summary['chemistry']['sa_median']:.3f}")
        if enable_vina and 'vina_dock_mean' in summary['chemistry']:
            logger.info(f"Vina Dock: mean={summary['chemistry']['vina_dock_mean']:.3f}, median={summary['chemistry']['vina_dock_median']:.3f}")
    
    # Save summary metrics to JSON file
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': summary['total_samples'],
        'failed_geometry_count': summary.get('failed_geometry_count', 0),
        'failed_chemistry_count': summary.get('failed_chemistry_count', 0),
        'evaluation_modules': summary['evaluation_modules']
    }
    
    # Add geometry metrics to JSON
    if enable_geom and 'geometry' in summary:
        summary_json['geometry_metrics'] = {
            'total_attempts': summary['geometry']['total_attempts'],
            'total_success': summary['geometry']['total_success'],
            'overall_success_rate': float(summary['geometry']['overall_success_rate']),
            'evaluation_success_rate': float(summary['geometry'].get('evaluation_success_rate', summary['geometry']['overall_success_rate'])),
            'quality_success_rate': float(summary['geometry'].get('quality_success_rate', 0.0)),
            'quality_pass_count': int(summary['geometry'].get('quality_pass_count', 0)),
            'fraction_mol_stable': float(summary['geometry']['fraction_mol_stable']),
            'fraction_atom_stable': float(summary['geometry']['fraction_atom_stable']),
            'intra_clash_atom_ratio': float(summary['geometry']['intra_clash_atom_ratio']),
            'inter_clash_atom_ratio': float(summary['geometry']['inter_clash_atom_ratio']),
            'clash_mol_ratio': float(summary['geometry']['clash_mol_ratio'])
        }
        
        # Add overall JSD metrics if available
        if 'overall_jsd_metrics' in summary['geometry']:
            jsd_metrics = summary['geometry']['overall_jsd_metrics']
            summary_json['geometry_metrics']['overall_jsd_metrics'] = {}
            
            if jsd_metrics.get('overall_bond_length_jsd') is not None:
                summary_json['geometry_metrics']['overall_jsd_metrics']['bond_length'] = {
                    'mean': float(jsd_metrics['overall_bond_length_jsd']),
                    'std': float(jsd_metrics['overall_bond_length_jsd_std'])
                }
            
            if jsd_metrics.get('overall_bond_angle_jsd') is not None:
                summary_json['geometry_metrics']['overall_jsd_metrics']['bond_angle'] = {
                    'mean': float(jsd_metrics['overall_bond_angle_jsd']),
                    'std': float(jsd_metrics['overall_bond_angle_jsd_std'])
                }
            
            if jsd_metrics.get('overall_torsion_angle_jsd') is not None:
                summary_json['geometry_metrics']['overall_jsd_metrics']['torsion_angle'] = {
                    'mean': float(jsd_metrics['overall_torsion_angle_jsd']),
                    'std': float(jsd_metrics['overall_torsion_angle_jsd_std'])
                }
    
    # Add chemistry metrics to JSON
    if enable_chem and 'chemistry' in summary:
        summary_json['chemistry_metrics'] = {
            'qed': {
                'mean': float(summary['chemistry']['qed_mean']),
                'median': float(summary['chemistry']['qed_median'])
            },
            'sa': {
                'mean': float(summary['chemistry']['sa_mean']),
                'median': float(summary['chemistry']['sa_median'])
            }
        }
        
        # Add Vina metrics if available
        if enable_vina and 'vina_dock_mean' in summary['chemistry']:
            summary_json['chemistry_metrics']['vina'] = {
                'score_only': {
                    'mean': float(summary['chemistry']['vina_score_only_mean']),
                    'median': float(summary['chemistry']['vina_score_only_median'])
                },
                'minimize': {
                    'mean': float(summary['chemistry']['vina_min_mean']),
                    'median': float(summary['chemistry']['vina_min_median'])
                },
                'dock': {
                    'mean': float(summary['chemistry']['vina_dock_mean']),
                    'median': float(summary['chemistry']['vina_dock_median'])
                }
            }
    
    # Save to JSON file
    summary_json_file = os.path.join(output_dir, 'evaluation_summary_metrics.json')
    with open(summary_json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"evaluation summary saved to: {summary_json_file}")

def collect_cif_parsing_statistics(cif_files, all_ligand_pocket_pairs, output_dir):
    """Collect and save CIF parsing statistics"""
    
    # Count total CIF files
    total_cifs = len(cif_files)
    
    # Count successful parsing
    successful_cifs = len(set([pair['cif_file'] for pair in all_ligand_pocket_pairs]))
    
    # Count failed CIF files
    failed_cifs = total_cifs - successful_cifs
    
    # Get failed CIF filenames
    successful_cif_files = set([pair['cif_file'] for pair in all_ligand_pocket_pairs])
    failed_cif_files = [cif for cif in cif_files if cif not in successful_cif_files]
    
    # Count total ligand-pocket pairs
    total_pairs = len(all_ligand_pocket_pairs)
    
    # Statistics
    statistics = {
        'timestamp': datetime.now().isoformat(),
        'cif_parsing_statistics': {
            'total_cif_files': total_cifs,
            'successful_cif_files': successful_cifs,
            'failed_cif_files': failed_cifs,
            'success_rate': successful_cifs / total_cifs if total_cifs > 0 else 0,
            'total_ligand_pocket_pairs': total_pairs,
            'average_pairs_per_cif': total_pairs / successful_cifs if successful_cifs > 0 else 0
        },
        'failed_cif_files': [os.path.basename(f) for f in failed_cif_files],
        'successful_cif_files': [os.path.basename(f) for f in successful_cif_files]
    }
    
    # Save statistics to JSON file
    stats_file = os.path.join(output_dir, 'cif_parsing_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    # Log statistics
    logger.info("=== CIF file parsing statistics ===")
    logger.info(f"total CIF files: {total_cifs}")
    logger.info(f"successful CIF files: {successful_cifs}")
    logger.info(f"failed CIF files: {failed_cifs}")
    logger.info(f"success rate: {statistics['cif_parsing_statistics']['success_rate']:.1%}")
    logger.info(f"total ligand-pocket pairs: {total_pairs}")
    logger.info(f"average pairs per CIF file: {statistics['cif_parsing_statistics']['average_pairs_per_cif']:.2f}")
    
    if failed_cifs > 0:
        logger.warning(f"failed CIF files: {failed_cifs} failed files")
        for failed_cif in failed_cif_files[:10]:  # Only show the first 10
            logger.warning(f"  - {os.path.basename(failed_cif)}")
        if failed_cifs > 10:
            logger.warning(f"  ... {failed_cifs - 10} more failed files")
        
    # Print to console as well
    logger.info("\n=== CIF file parsing statistics ===")
    logger.info(f"Total CIF files: {total_cifs}")
    logger.info(f"successful CIF files: {successful_cifs}")
    logger.info(f"failed CIF files: {failed_cifs}")
    logger.info(f"success rate: {statistics['cif_parsing_statistics']['success_rate']:.1%}")
    logger.info(f"total ligand-pocket pairs: {total_pairs}")
    logger.info(f"average ligand-pocket pairs per CIF file: {statistics['cif_parsing_statistics']['average_pairs_per_cif']:.2f}")
    
    if failed_cifs > 0:
        logger.warning(f"\nFailed CIF files ({failed_cifs}):")
        for failed_cif in failed_cif_files[:10]:  # Only show the first 10
            logger.warning(f"  - {os.path.basename(failed_cif)}")
        if failed_cifs > 10:
            logger.warning(f"  ... {failed_cifs - 10} more failed files")
    
    return statistics


def collect_dataset_statistics(ligand_pocket_pairs, evaluation_results, output_dir):
    """Collect and save dataset statistics"""
    
    # Filter valid evaluation results
    valid_results = [r for r in evaluation_results if r is not None and r['geom_result'] is not None and r['chem_result'] is not None]
    
    # Statistics from ligand-pocket pairs
    total_pocket_ligand_pairs = len(ligand_pocket_pairs)
    
    # Get ligand identifiers and SMILES for unique counting
    ligand_identifiers = [pair['ligand_identifier'] for pair in ligand_pocket_pairs]
    total_ligands = len(ligand_identifiers)
    
    # Count unique ligands by identifier
    unique_ligands_by_identifier = len(set(ligand_identifiers))
    
    # Count unique ligands by SMILES (more accurate)
    unique_smiles = set()
    smiles_to_identifier = {}
    
    for result in valid_results:
        if result['chem_result'] and 'smiles' in result['chem_result']:
            smiles = result['chem_result']['smiles']
            ligand_file = result['ligand_file']
            
            # Extract ligand identifier from filename
            ligand_identifier = os.path.basename(ligand_file).replace('_ligand.sdf', '')
            
            unique_smiles.add(smiles)
            if smiles not in smiles_to_identifier:
                smiles_to_identifier[smiles] = []
            smiles_to_identifier[smiles].append(ligand_identifier)
    
    unique_ligands_by_smiles = len(unique_smiles)
    
    # Get ligand name statistics
    ligand_names = [pair['ligand_identifier'].split('_')[0] for pair in ligand_pocket_pairs]
    ligand_name_counts = Counter(ligand_names)
    
    # Create comprehensive statistics
    statistics = {
        'timestamp': datetime.now().isoformat(),
        'dataset_statistics': {
            'total_pocket_ligand_pairs': total_pocket_ligand_pairs,
            'total_ligands': total_ligands,
            'unique_ligands_by_identifier': unique_ligands_by_identifier,
            'unique_ligands_by_smiles': unique_ligands_by_smiles,
            'successfully_evaluated_pairs': len(valid_results),
            'evaluation_success_rate': len(valid_results) / total_pocket_ligand_pairs if total_pocket_ligand_pairs > 0 else 0
        },
        'ligand_name_distribution': dict(ligand_name_counts.most_common(20)),  # Top 20 most common ligand names
        'smiles_to_identifier_mapping': smiles_to_identifier
    }
    
    # Save statistics to JSON file
    stats_file = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    # Log statistics
    logger.info("=== Dataset statistics ===")
    logger.info(f"Total ligand-pocket pairs: {total_pocket_ligand_pairs}")
    logger.info(f"Total ligands: {total_ligands}")
    logger.info(f"Unique ligands by identifier: {unique_ligands_by_identifier}")
    logger.info(f"Unique ligands by SMILES: {unique_ligands_by_smiles}")
    logger.info(f"Successfully evaluated ligand-pocket pairs: {len(valid_results)}")
    logger.info(f"Evaluation success rate: {len(valid_results) / total_pocket_ligand_pairs * 100:.1f}%")
    logger.info(f"Most common ligand names: {dict(ligand_name_counts.most_common(5))}")
    
    # Print to console as well
    logger.info("\n=== Dataset statistics ===")
    logger.info(f"Total ligand-pocket pairs: {total_pocket_ligand_pairs}")
    logger.info(f"Total ligands: {total_ligands}")
    logger.info(f"Unique ligands by identifier: {unique_ligands_by_identifier}")
    logger.info(f"Unique ligands by SMILES: {unique_ligands_by_smiles}")
    logger.info(f"Successfully evaluated ligand-pocket pairs: {len(valid_results)}")
    logger.info(f"Evaluation success rate: {len(valid_results) / total_pocket_ligand_pairs * 100:.1f}%")
    logger.info(f"Most common ligand names: {dict(ligand_name_counts.most_common(5))}")
    
    return statistics


def setup_logging(output_dir, verbose=False):
    """Setup logging configuration"""
    # Create logs directory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f'pipeline_{timestamp}.log')
    
    # Set logging level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure global logger  
    global logger
    logger.info("=== Ligand-pocket analysis pipeline started ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {'DEBUG' if verbose else 'INFO'}")

    # Configure RDKit logging
    rdBase.LogToPythonLogger()
    file_handler = logging.FileHandler(os.path.join(logs_dir, f'pipeline_{timestamp}_rdkit.log'))
    formatter = logging.Formatter("[RDKit]: %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    rdkit_logger.addHandler(file_handler)
    rdkit_logger.setLevel(logging.DEBUG)
    return logger,rdkit_logger


def main():
    parser = argparse.ArgumentParser(description='Full pipeline for ligand-pocket analysis')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing CIF files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--dist_cutoff', type=float, default=10.0, help='Distance cutoff for pocket definition')
    parser.add_argument('--center', type=float, nargs=3, default=None, help='Center coordinates for docking')
    parser.add_argument('--exhaustiveness', type=int, default=16, help='Exhaustiveness for docking')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes (default: CPU count)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Add evaluation module switches
    parser.add_argument('--enable_geom', action='store_true', default=True, 
                        help='Enable geometry evaluation (default: True)')
    parser.add_argument('--disable_geom', action='store_true', 
                        help='Disable geometry evaluation')
    parser.add_argument('--enable_chem', action='store_true', default=True,
                        help='Enable chemistry evaluation (default: True)')
    parser.add_argument('--disable_chem', action='store_true',
                        help='Disable chemistry evaluation')
    parser.add_argument('--enable_vina', action='store_true', default=True,
                        help='Enable Vina docking evaluation (default: True)')
    parser.add_argument('--disable_vina', action='store_true',
                        help='Disable Vina docking evaluation')
    
    # Add CCD distribution file path arguments
    parser.add_argument('--ccd_bond_length_path', type=str, default=None,
                        help='Path to CCD bond length distribution file')
    parser.add_argument('--ccd_bond_angle_path', type=str, default=None,
                        help='Path to CCD bond angle distribution file')
    parser.add_argument('--ccd_torsion_angle_path', type=str, default=None,
                        help='Path to CCD torsion angle distribution file')
    
    args = parser.parse_args()
    
    # Handle enable/disable flags
    if args.disable_geom:
        args.enable_geom = False
    if args.disable_chem:
        args.enable_chem = False
    if args.disable_vina:
        args.enable_vina = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger,rdkit_logger = setup_logging(args.output_dir, args.verbose)

    # Set CCD distribution file paths if provided
    if args.ccd_bond_length_path:
        set_ccd_bond_length_path(args.ccd_bond_length_path)
        logger.info(f"Set CCD bond length distribution file path: {args.ccd_bond_length_path}")
    if args.ccd_bond_angle_path:
        set_ccd_bond_angle_path(args.ccd_bond_angle_path)
        logger.info(f"Set CCD bond angle distribution file path: {args.ccd_bond_angle_path}")
    if args.ccd_torsion_angle_path:
        set_ccd_torsion_angle_path(args.ccd_torsion_angle_path)
        logger.info(f"Set CCD torsion angle distribution file path: {args.ccd_torsion_angle_path}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Distance cutoff: {args.dist_cutoff}")
    logger.info(f"Exhaustiveness: {args.exhaustiveness}")
    logger.info(f"Evaluation modules - Geometry: {args.enable_geom}, Chemistry: {args.enable_chem}, Vina: {args.enable_vina}")
    
    # Check if at least one evaluation module is enabled
    if not (args.enable_geom or args.enable_chem):
        logger.error("At least one evaluation module (geometry or chemistry) needs to be enabled")
        return

    # Set number of processes
    if args.num_processes is None:
        args.num_processes = cpu_count()
    
    logger.info(f"Using {args.num_processes} processes for processing")
    
    # Find all CIF files
    cif_files = glob(os.path.join(args.input_dir, '*.cif'))
    if not cif_files:
        logger.error(f"No CIF files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(cif_files)} CIF files")
    
    # Step 1: Process CIF files to extract ligand-pocket pairs
    logger.info("=== Step 1: Process CIF files to extract ligand-pocket pairs ===")
    
    # Create subdirectory for parsed files
    parsed_dir = os.path.join(args.output_dir, 'parsed_structures')
    os.makedirs(parsed_dir, exist_ok=True)
    
    all_ligand_pocket_pairs = []

    # single process
    # for cif_file in cif_files:
    #     results = process_single_cif(cif_file, parsed_dir, args.dist_cutoff)

    #     for result in results:
    #         all_ligand_pocket_pairs.append(result)

    with Pool(processes=args.num_processes) as pool:
        cif_args = [(cif_file, parsed_dir, args.dist_cutoff) for cif_file in cif_files]
        results = pool.starmap(process_single_cif, cif_args)
        
        for result in results:
            all_ligand_pocket_pairs.extend(result)
    
    logger.info(f"Successfully extracted {len(all_ligand_pocket_pairs)} ligand-pocket pairs")
    
    # Exit(0)

    if not all_ligand_pocket_pairs:
        logger.error("No valid ligand-pocket pairs found")
        return
    
    # Step 2: Evaluate geometry and chemistry
    enabled_modules = []
    if args.enable_geom:
        enabled_modules.append("Geometry")
    if args.enable_chem:
        enabled_modules.append("Chemistry")
    if args.enable_vina:
        enabled_modules.append("Vina Docking")
    
    logger.info(f"=== Step 2: Evaluate {', '.join(enabled_modules)} properties ===")
    
    # Prepare arguments for multiprocessing
    eval_args = []
    for pair in all_ligand_pocket_pairs:
        eval_args.append((
            pair['ligand_file'],
            pair['pocket_file'], 
            args.output_dir,
            args.center,
            args.exhaustiveness,
            args.enable_geom,
            args.enable_chem,
            args.enable_vina
        ))
    
    # single process
    # evaluation_results = []
    # for eval_arg in eval_args:
    #     evaluation_results.append(process_single_ligand_pocket_pair(eval_arg))
    
    # Run evaluations in parallel
    with Pool(processes=args.num_processes) as pool:
        evaluation_results = pool.map(process_single_ligand_pocket_pair, eval_args)
    
    # Step 3: Collect and save results
    logger.info("=== Step 3: Collect and save results ===")
    collect_and_save_results(evaluation_results, args.output_dir, args.enable_geom, args.enable_chem, args.enable_vina)
    
    # # Step 4: Collect and save dataset statistics
    # logger.info("=== Step 4: Collect and save dataset statistics ===")
    
    # Step 5: Collect and save CIF parsing statistics
    logger.info("=== Step 5: Collect and save CIF parsing statistics ===")
    collect_cif_parsing_statistics(cif_files, all_ligand_pocket_pairs, args.output_dir)

    logger.info("=== Pipeline processing completed ===")
    logger.info(f"Pipeline processing completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
