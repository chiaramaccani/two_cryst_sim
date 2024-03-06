import json
import numpy as np
from pathlib import Path
import sys
import os
import yaml
import pandas as pd
import pickle

import xobjects as xo
import xtrack as xt
import xpart as xp
import xcoll as xc
import scipy
import io 

from IPython import embed

def main():

    # Get the input file
    input_collimator_file = 'CollDB_HL_tight_b4.data' #sys.argv[1]
    output_collimator_file = 'HL_tight_twocryst' #sys.argv[2]
    emitt = 2.5e-6

    colldb = xc.CollimatorDatabase.from_SixTrack(input_collimator_file, nemitt_x=emitt, nemitt_y=emitt, ignore_crystals=False)
    colldb.write_to_yaml(output_collimator_file)

    


if __name__ == "__main__":
    main()
