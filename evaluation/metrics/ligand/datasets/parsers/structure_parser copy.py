import Bio.PDB
from Bio.PDB import MMCIFIO, PDBIO, Select
from pathlib import Path
import scipy.spatial
import string
import logging
import os
import traceback
from rdkit import Chem

# import ccd constants
from metrics.ligand.utils.molecule.constants import metal_ions, crystallographic_additives
from metrics.ligand.utils.protein.constants import standard_aa, modified_aa, nucleic_acids

# breakpoint()s

logger = logging.getLogger(__name__)
rdkit_logger = logging.getLogger('rdkit')

# breakpoint()
def fix_invalid_chain_ids(structure):
    """Fix invalid chain IDs that exceed PDB format limit (single character)"""
    # PDB format only supports single-character chain_ids, need to modify if exceeded
    valid_chain_chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    used_chain_ids = set()
    chain_id_mapping = {}
    
    # First, collect all existing chain_ids
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if len(chain_id) > 1:  # Invalid chain_id
                # Find available single-character chain_id
                new_chain_id = None
                for char in valid_chain_chars:
                    if char not in used_chain_ids:
                        new_chain_id = char
                        used_chain_ids.add(char)
                        break
                
                if new_chain_id is None:
                    # If all single characters are used up, use numbers and special characters
                    for i in range(10):
                        char = str(i)
                        if char not in used_chain_ids:
                            new_chain_id = char
                            used_chain_ids.add(char)
                            break
                
                if new_chain_id is None:
                    # Last resort: use special characters
                    special_chars = "!@#$%^&*()-_=+[]{}|;:,.<>?"
                    for char in special_chars:
                        if char not in used_chain_ids:
                            new_chain_id = char
                            used_chain_ids.add(char)
                            break
                
                if new_chain_id is None:
                    # If still not found, use first available character, may have conflicts
                    new_chain_id = 'X'
                
                chain_id_mapping[chain_id] = new_chain_id
                logger.info(f"Modified invalid chain_id: '{chain_id}' -> '{new_chain_id}'")
                
                # Modify chain id
                chain.id = new_chain_id
            else:
                used_chain_ids.add(chain_id)
    
    return structure, chain_id_mapping

class PocketSelect(Select):
    """Selection class for pocket residues"""
    def __init__(self, pocket_residues):
        self.pocket_residues = pocket_residues
    
    def accept_residue(self, residue):
        return residue in self.pocket_residues

# TODO: Change to use -L as residue
def get_ligand_residues(structure):
    """Extract all ligand residues from structure, excluding modified amino acids and nucleic acids"""
    exclude_residues = set(standard_aa + modified_aa + metal_ions + crystallographic_additives + nucleic_acids + ['HOH'])

    ligand_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                if resname not in exclude_residues:
                    ligand_residues.append(residue)

    return ligand_residues

def get_pocket_residues(structure, ligand_residue, dist_cutoff):
    """Get pocket residues within distance cutoff from ligand"""
    protein_atoms = []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        if residue.get_resname() in standard_aa + modified_aa:
            protein_atoms.append(atom)
    
    protein_coords = [atom.get_coord() for atom in protein_atoms]
    ligand_coords = [atom.get_coord() for atom in ligand_residue.get_atoms()]

    kd_tree = scipy.spatial.KDTree(protein_coords)
    nearby_indices = kd_tree.query_ball_point(ligand_coords, r=dist_cutoff, p=2.0)
    nearby_indices = set([k for l in nearby_indices for k in l])
    
    pocket_residues = set()
    for i in nearby_indices:
        atom = protein_atoms[i]
        residue = atom.get_parent()
        if residue.get_resname() != 'HOH':
            pocket_residues.add(residue)
    
    return pocket_residues

def ligand_to_sdf(ligand_residue, output_file, cif_file=None):
    """Convert ligand residue to SDF format using Bio.PDB (fallback method)"""
    # Create temporary PDB file in the same directory as output_file
    # Original code:
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
    #     tmp_pdb_path = tmp_pdb.name
    output_dir = os.path.dirname(output_file)
    output_basename = os.path.splitext(os.path.basename(output_file))[0]
    tmp_pdb_path = os.path.join(output_dir, f"{output_basename}_tmp.pdb")
    
    try:
        # Manually create PDB standard compliant file instead of using Bio.PDB's PDBIO
        lines = []
        lines.append("HEADER    LIGAND\n")
        lines.append("TITLE     LIGAND FROM CIF\n")
        
        atom_serial = 1
        for atom in ligand_residue.get_atoms():
            # Get atom information
            atom_name = atom.get_name()
            coord = atom.get_coord()
            occupancy = atom.get_occupancy()
            bfactor = atom.get_bfactor()
            
            # Infer element
            element = atom.element if hasattr(atom, 'element') else atom_name[0]
            
            # Format strictly according to PDB format standard
            line = "HETATM"  # Columns 1-6
            line += f"{atom_serial:5d}"  # Columns 7-11 (atom serial number)
            line += " "  # Column 12 (space)
            line += f"{atom_name:<4s}"  # Columns 13-16 (atom name, left-aligned)
            line += " "  # Column 17 (alternate location indicator)
            line += f"{ligand_residue.get_resname():<3s}"  # Columns 18-20 (residue name)
            line += " "  # Column 21 (space)
            line += f"{ligand_residue.get_parent().get_id()}"  # Column 22 (chain identifier)
            line += f"{ligand_residue.get_id()[1]:4d}"  # Columns 23-26 (residue sequence number)
            line += " "  # Column 27 (insertion code)
            line += "   "  # Columns 28-30 (space)
            line += f"{coord[0]:8.3f}"  # Columns 31-38 (X coordinate)
            line += f"{coord[1]:8.3f}"  # Columns 39-46 (Y coordinate)
            line += f"{coord[2]:8.3f}"  # Columns 47-54 (Z coordinate)
            line += f"{occupancy:6.2f}"  # Columns 55-60 (occupancy)
            line += f"{bfactor:6.2f}"  # Columns 61-66 (B-factor)
            line += "          "  # Columns 67-76 (space)
            line += f"{element:>2s}"  # Columns 77-78 (element symbol)
            line += "\n"
            
            lines.append(line)
            atom_serial += 1
        
        lines.append("END\n")
        
        # Write manually created pdb file
        with open(tmp_pdb_path, 'w') as f:
            f.writelines(lines)
        
        
        rdkit_logger.info(f'convert ligand to sdf Chem.MolFromPDBFile {tmp_pdb_path} to sdf:...')
        mol = Chem.MolFromPDBFile(tmp_pdb_path, removeHs=False)

        if mol is not None and mol.GetNumAtoms() > 0:
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()
            return True
        else:
            cif_info = f"original cif file: {cif_file}" if cif_file else f"temporary pdb file: {tmp_pdb_path}"
            logger.warning(f"warning: cannot convert ligand {tmp_pdb_path} {ligand_residue.get_resname()} to sdf, {cif_info}")
            return False
            
    except Exception as e:
        logger.error(f"error: convert ligand to sdf failed: {e}")
        return False
    
    # finally:
    #     if os.path.exists(tmp_pdb_path):
    #         os.remove(tmp_pdb_path)

def process_single_cif(cif_file, output_dir, dist_cutoff=10.0):
    """Process a single CIF file to extract ligand-pocket pairs"""

    # breakpoint()
    try:
        parser = Bio.PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure('complex', cif_file)
        
        # fix chain id
        structure, chain_id_mapping = fix_invalid_chain_ids(structure)
        if chain_id_mapping:
            logger.info(f"fixed {len(chain_id_mapping)} invalid chain ids")
        
        base_name = Path(cif_file).stem
        ligand_residues = get_ligand_residues(structure)

        # breakpoint()
        
        if not ligand_residues:
            logger.info(f"no valid ligand found in {cif_file}")
            return []
        
        logger.info(f"found {len(ligand_residues)} ligands in {cif_file}")
        
        results = []
        
        for i, ligand_residue in enumerate(ligand_residues):
            ligand_name = ligand_residue.get_resname()
            ligand_chain = ligand_residue.get_parent().get_id()
            ligand_id = ligand_residue.get_id()[1]
            
            ligand_identifier = f"{ligand_name}_{ligand_chain}_{ligand_id}"
            
            logger.info(f"processing ligand {i+1}/{len(ligand_residues)}: {ligand_identifier}")
            
            # Get pocket residues
            pocket_residues = get_pocket_residues(structure, ligand_residue, dist_cutoff)
            
            if not pocket_residues:
                logger.info(f"no pocket residues found for ligand {ligand_identifier}")
                continue
            
            # Save pocket as PDB
            pocket_file = os.path.join(output_dir, f"{base_name}_{ligand_identifier}_pocket.pdb")
            io = PDBIO()
            io.set_structure(structure)
            io.save(pocket_file, PocketSelect(pocket_residues))
            
            # Save ligand as SDF
            ligand_file = os.path.join(output_dir, f"{base_name}_{ligand_identifier}_ligand.sdf")
            success = ligand_to_sdf(ligand_residue, ligand_file, cif_file)
            
            if success:
                results.append({
                    'cif_file': cif_file,
                    'ligand_identifier': ligand_identifier,
                    'pocket_file': pocket_file,
                    'ligand_file': ligand_file,
                    'pocket_residues_count': len(pocket_residues)
                })
                logger.info(f"successfully saved: {ligand_identifier}")
            else:
                logger.info(f"skip invalid ligand: {ligand_identifier}")
        
        return results
        
    except Exception as e:
        logger.error(f"error: process cif file {cif_file} failed: {e}")
        logger.error(traceback.format_exc())
        return []