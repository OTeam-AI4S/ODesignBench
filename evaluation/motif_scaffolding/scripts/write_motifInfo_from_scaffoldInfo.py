#!/usr/bin/env python3
import sys
import csv
import os

def read_pdb_residues(pdb_file):
    """Read residues from a PDB file and organize them by chain."""
    residues = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                chain_id = line[21].strip()
                res_num = int(line[22:26])
                res_name = line[17:20].strip()
                residues.setdefault(chain_id, {})[res_num] = res_name
    return residues

def get_residue_ranges(res_nums):
    """Convert a list of residue numbers into ranges."""
    ranges = []
    sorted_nums = sorted(set(res_nums))
    if not sorted_nums:
        return ranges
    start = prev = sorted_nums[0]
    for num in sorted_nums[1:]:
        if num == prev + 1:
            prev = num
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = num
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ranges

def format_redesign_positions(chain_id, res_nums):
    """Format redesign positions for a chain into a string."""
    ranges = get_residue_ranges(res_nums)
    return ';'.join(f"{chain_id}{r}" for r in ranges)

def main():
    if len(sys.argv) != 4:
        print("Usage: ./write_motifInfo_from_scaffoldInfo.py scaffold_info.csv motif.pdb test_cases.csv")
        sys.exit(1)

    scaffold_info_file = sys.argv[1]
    pdb_file = sys.argv[2]
    output_file = sys.argv[3]
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]

    # Read residues from the motif PDB file
    residue_data = read_pdb_residues(pdb_file)

    # Collect redesign positions where residue type is 'UNK'
    redesign_positions = []
    for chain_id, residues in residue_data.items():
        unk_res_nums = [res_num for res_num, res_name in residues.items() if res_name == 'UNK']
        if unk_res_nums:
            chain_ranges = format_redesign_positions(chain_id, unk_res_nums)
            redesign_positions.append(chain_ranges)

    # Prepare redesign_positions string
    redesign_positions_str = ';'.join(redesign_positions)

    # Read scaffold_info.csv and write to motif_info.csv
    with open(scaffold_info_file, 'r', newline='') as csv_infile, \
         open(output_file, 'w', newline='') as csv_outfile:

        reader = csv.DictReader(csv_infile)
        fieldnames = ['pdb_name', 'sample_num', 'contig', 'redesign_positions', 'segment_order']
        writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            sample_num = row['sample_num'].strip()
            motif_placements = row['motif_placements'].strip()

            # Parse motif_placements into numbers and letters
            parts = motif_placements.strip('/').split('/')

            # Build segment_order
            segment_order = ';'.join([part[0] for part in parts if
                part[0].isalpha()])

            # Build contig
            contig_parts = []
            for part in parts:
                if part[0].isalpha():
                    chain = part[0]
                    res_nums = sorted(residue_data.get(chain, {}).keys())
                    res_ranges = get_residue_ranges(res_nums)
                    range_str = ','.join(f"{chain}{r}" for r in res_ranges)
                    contig_parts.append(range_str)
                else:
                    contig_parts.append(part)
            contig = '/'.join(contig_parts)

            # Write to motif_info.csv
            writer.writerow({
                'pdb_name': pdb_name,
                'sample_num': sample_num,
                'contig': contig,
                'redesign_positions': redesign_positions_str,
                'segment_order': segment_order
            })

if __name__ == "__main__":
    main()

