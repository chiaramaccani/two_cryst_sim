import numpy as np
import json
import matplotlib.pyplot as plt
import sys


import xobjects as xo
import xtrack as xt
import xpart as xp

from cpymad.madx import Madx


file_name = sys.argv[1]
out_name = sys.argv[2]
print(file_name)
mad = Madx()
mad.call(str(file_name))
line = xt.Line.from_madx_sequence(mad.sequence['lhcb2'], install_apertures=True)
line.particle_ref = xp.Particles(q0=1, mass0=xp.PROTON_MASS_EV, gamma0=mad.sequence['lhcb2'].beam.gamma)
line.to_json(str(out_name))