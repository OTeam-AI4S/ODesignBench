import os
import sys
import tempfile
import pandas as pd
from pathlib import Path
from collections import defaultdict
# debug
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "USalign"))
# sys.path.append('evaluation/metrics/USalign')

# Get the USalign directory path
USALIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'USalign')
TMSCORE_PATH = os.path.join(USALIGN_DIR, 'TMscore')
QTMCLUST_PATH = os.path.join(USALIGN_DIR, 'qTMclust')

class USalign:

    def __init__(self):
        pass
    
    @staticmethod
    def compute_tmscore(pred: str, refold: str):
        
        fd, path = tempfile.mkstemp(suffix=".txt", text=True)
        cmd = f'{TMSCORE_PATH} {pred} {refold} -outfmt 2 > {path}'
        os.system(cmd)
        df = pd.read_csv(path, delimiter='\t')
        df = df[['#PDBchain1', "PDBchain2", "TM2", "Lali"]]
        tmscore = df['TM2'].values.tolist()[0]
        os.remove(path)

        return tmscore

    @staticmethod
    def compute_qTMclust_metrics(success_dir: str, gen_dir: str, tm_thresh: int):

        n_success = 0
        n_cluster = 0
        diversity = {}
        ret_path = os.path.join(os.path.dirname(success_dir), "MASTER_QTM_RESULTS.txt")
        for success_list in Path(success_dir).glob("*.list"):
            # breakpoint()
            CMD = f"{QTMCLUST_PATH} -dir {gen_dir} {str(success_list)} -TMcut {tm_thresh} -o {ret_path}"
            os.system(CMD)
            with open(str(success_list), 'r+') as f:
                n = len(f.readlines())
            n_success += n
            with open(ret_path, "r+") as f:
                clusters = f.readlines()
            n_cluster += len(clusters)
            diversity[os.path.basename(success_list)] = len(clusters) / n
        # breakpoint()
        diversity['avg'] = sum(diversity.values()) / len(diversity)
        print(f"n_success: {n_success}, n_cluster: {n_cluster}, diversity: {diversity['avg']}")
        os.remove(ret_path)
        df = pd.DataFrame.from_dict(diversity, orient='index', columns=['diversity'])
        df.to_csv(os.path.join(os.path.dirname(success_dir), "diversity.csv"), index=True)
        
        