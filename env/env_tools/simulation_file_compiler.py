import os
import sys

import json
import numpy as np
from tqdm import tqdm


def sim_PWD_compiler(
    SAW_case = None,
    sim_file = None): # TODO
    """
    Description:
        Read PWD file and then get basic network information
    Parameters:
        sim_file: string, the PWD file location
    Returns:
        net_topology: array
    """
    if SAW_case is None:
        SAW_case = sim_file
    return  