import numpy as np
import torch

from .protein_parser import *
from .structure_parser import *

try:
    from .molecule_parser import *
    _MOLECULE_PARSER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    # Some evaluation paths only need structure parsing. Keep those working even
    # if optional chemistry decomposition dependencies are absent.
    _MOLECULE_PARSER_IMPORT_ERROR = exc


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        elif isinstance(v, list):
            if isinstance(v[0], np.ndarray):
                output[k] = [torch.from_numpy(v[ii]) for ii in range(len(v))]
            else:
                output[k] = v
        else:
            output[k] = v
    return output
