### AME `mcsa_41` assets

- **`RFD2_example.json`**: structured metadata per case.
  - **`case_id`**: e.g. `M0630_1j79`
  - **`reference_pdb`**: path to the native reference complex PDB (relative to this folder)
  - **`ligands`**: ligand / metal residue names (as used in the reference PDB)
  - **`motif_contig_atoms`**: *the “atomic motif” definition* from RFdiffusion2:
    - mapping **`<chain><resid>` → `[atom_name, ...]`**
    - example: `{"A250": ["OD1","CG"]}`
  - **`partially_fixed_ligand_atoms`**: ligand atoms that were fixed/used by RFdiffusion2 (if present)
  - **`contigs`**: RFdiffusion2 contig strings

- **`reference_pdbs/`**: the 41 **native enzyme–ligand complex** PDBs used by `mcsa_41`.

