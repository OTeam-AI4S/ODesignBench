import os
import json
import subprocess
from typing import Dict


class FoldSeek():
    def __init__(self, config):
        self.config = config
        self.foldseek_bin = config.get('foldseek_bin', 'foldseek')
        self.database_dir = config.get('foldseek_database', None)
        self.verbose = config.get('verbose', False)
        
        # Debug: print initialization
        if self.verbose:
            print(f"FoldSeek initialized - bin: {self.foldseek_bin}, database: {self.database_dir}")

    def compute_diversity(self, structure_dir: str, output_dir: str, dump: bool = True) -> Dict:
        """
        Calculate structural diversity using foldseek clustering
        
        Args:
            structure_dir: Directory containing structure files (.pdb)
            output_dir: Directory to save clustering results
        
        Returns:
            Dictionary with diversity metrics
        """
        print("=== Computing Structural Diversity ===")
        
        # Check if structure directory exists and has structures
        if not os.path.exists(structure_dir):
            print(f"✗ Structure directory not found: {structure_dir}")
            return {'diversity': 0.0, 'num_clusters': 0, 'num_structures': 0}
        
        # Count structures
        structures = [f for f in os.listdir(structure_dir) if f.endswith('.pdb')]
        num_structures = len(structures)
        
        print(f"Found {num_structures} structures")
        
        if num_structures == 0:
            return {'diversity': 0.0, 'num_clusters': 0, 'num_structures': 0}
        
        if num_structures == 1:
            return {'diversity': 1.0, 'num_clusters': 1, 'num_structures': 1}
        
        # Setup clustering directory
        cluster_dir = os.path.join(output_dir, "clustering")
        os.makedirs(cluster_dir, exist_ok=True)
        
        cluster_result = os.path.join(cluster_dir, "cluster")
        tmp_dir = os.path.join(cluster_dir, "tmp")
        
        # Run foldseek clustering
        cmd = [
            self.foldseek_bin, "easy-cluster",
            structure_dir, cluster_result, tmp_dir,
            "--min-seq-id", "0",
            "--alignment-type", "1", 
            "--cov-mode", "0",
            "--tmscore-threshold", "0.6"
        ]
        
        print(f"Running clustering command: {' '.join(cmd)}")
        
        try:
            cluster_file = f"{cluster_result}_cluster.tsv"
            if not os.path.exists(cluster_file):
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if self.verbose:
                    print("Clustering stdout:", result.stdout)
            else:
                print(f"✓ Clustering results file already exists: {cluster_file}")
            
            # Read clustering results
            if os.path.exists(cluster_file):
                cluster_representatives = set()
                with open(cluster_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            cluster_representatives.add(parts[0])
                
                num_clusters = len(cluster_representatives)
                diversity = num_clusters / num_structures
                
                print(f"✓ Clustering completed: {num_clusters} clusters, diversity: {diversity:.4f}")
                
                result = {
                    'diversity': diversity,
                    'num_clusters': num_clusters,
                    'num_structures': num_structures
                }
                
                # Save results if dump is True
                if dump:
                    os.makedirs(output_dir, exist_ok=True)
                    result_file = os.path.join(output_dir, "diversity_results.json")
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"✓ Results saved to {result_file}")
                
                return result
            else:
                print("✗ Clustering results file not found")
                return {'diversity': 0.0, 'num_clusters': 0, 'num_structures': num_structures}
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Clustering failed: {e}")
            if self.verbose:
                print("STDERR:", e.stderr)
            return {'diversity': 0.0, 'num_clusters': 0, 'num_structures': num_structures}

    def compute_novelty(self, structure_dir: str, output_dir: str, dump: bool = True) -> Dict:
        """
        Calculate structural novelty based on max TM-score average against database
        
        Args:
            structure_dir: Directory containing structure files (.pdb)
            output_dir: Directory to save search results
        
        Returns:
            Dictionary with novelty metrics
        """
        print("=== Computing Structural Novelty ===")
        
        # Check if structure directory exists
        if not os.path.exists(structure_dir):
            print(f"✗ Structure directory not found: {structure_dir}")
            return {'novelty': 0.0, 'max_tmscore_avg': 0.0, 'num_structures': 0}
        
        # Check if database is configured
        if not self.database_dir:
            print("✗ Foldseek database not configured")
            print(f"   Config dict keys: {list(self.config.keys())}")
            print(f"   database_dir value: {self.database_dir}")
            return {'novelty': 0.0, 'max_tmscore_avg': 0.0, 'num_structures': 0}
        
        # Check if database file exists
        if not os.path.exists(self.database_dir):
            print(f"✗ Foldseek database file not found: {self.database_dir}")
            return {'novelty': 0.0, 'max_tmscore_avg': 0.0, 'num_structures': 0}
        
        # Count structures
        structures = [f for f in os.listdir(structure_dir) if f.endswith('.pdb')]
        num_structures = len(structures)
        
        print(f"Found {num_structures} structures")
        
        if num_structures == 0:
            return {'novelty': 0.0, 'max_tmscore_avg': 0.0, 'num_structures': 0}
        
        if num_structures == 1:
            return {'novelty': 1.0, 'max_tmscore_avg': 0.0, 'num_structures': 1}
        
        # Setup search directory
        search_dir = os.path.join(output_dir, "novelty_search")
        os.makedirs(search_dir, exist_ok=True)
        
        search_result = os.path.join(search_dir, "search_results")
        tmp_dir = os.path.join(search_dir, "tmp")
        
        # Run foldseek easy-search for all-vs-all comparison
        cmd = [
            self.foldseek_bin, "easy-search",
            structure_dir, self.database_dir, search_result, tmp_dir,
            "--alignment-type", "1", 
            "--exhaustive-search",
            "--tmscore-threshold", "0.0",
            "--max-seqs", "10000000000",
            "--format-output", "query,target,alntmscore,lddt"
        ]
        
        print(f"Running novelty search command: {' '.join(cmd)}")
        
        try:
            if not os.path.exists(search_result):
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if self.verbose:
                    print("Novelty search stdout:", result.stdout)
            else:
                print(f"✓ Novelty search results file already exists: {search_result}")
            
            # Read search results and calculate max TM-score for each structure
            search_file = f"{search_result}"
            if os.path.exists(search_file):
                # Parse results and build TM-score matrix
                tm_scores = {}
                with open(search_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            query = parts[0]
                            target = parts[1]
                            tm_score = float(parts[2])
                            
                            # Skip self-comparisons
                            if query != target:
                                if query not in tm_scores:
                                    tm_scores[query] = []
                                tm_scores[query].append(tm_score)
                
                # Calculate max TM-score for each structure and take average
                structures_without_ext = [os.path.splitext(s)[0] for s in structures]
                max_tm_scores = []
                for struct_name in structures_without_ext:
                    if struct_name in tm_scores and tm_scores[struct_name]:
                        max_tm = max(tm_scores[struct_name])
                        max_tm_scores.append(max_tm)
                        if self.verbose:
                            print(f"Structure {struct_name}: max TM-score = {max_tm:.4f}")
                
                if max_tm_scores:
                    max_tmscore_avg = sum(max_tm_scores) / len(max_tm_scores)
                    # Novelty is defined as 1 - average max TM-score
                    novelty = 1.0 - max_tmscore_avg
                else:
                    max_tmscore_avg = 0.0
                    novelty = 1.0
                
                print(f"✓ Novelty calculation completed: avg max TM-score = {max_tmscore_avg:.4f}, novelty = {novelty:.4f}")
                
                result = {
                    'novelty': novelty,
                    'max_tmscore_avg': max_tmscore_avg,
                    'num_structures': num_structures
                }
                
                # Save results if dump is True
                if dump:
                    os.makedirs(output_dir, exist_ok=True)
                    result_file = os.path.join(output_dir, "novelty_results.json")
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"✓ Results saved to {result_file}")
                
                return result
            else:
                print("✗ Novelty search results file not found")
                return {'novelty': 0.0, 'max_tmscore_avg': 0.0, 'num_structures': num_structures}
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Novelty search failed: {e}")
            if self.verbose:
                print("STDERR:", e.stderr)
            return {'novelty': 0.0, 'max_tmscore_avg': 0.0, 'num_structures': num_structures}