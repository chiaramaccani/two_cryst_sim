import json
import sys
from pathlib import Path
import numpy as np
import os
from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
import xpart as xp


#input_file  = Path(str(sys.argv[1]))
#beam        = 1  # int(sys.argv[2])
beam        = 2  # int(sys.argv[2])
#line_name = os.path.expandvars("${HOME_TWOCRYST}/lossmaps_ALADDIN/opt_round_150_1500_optphases_b1.json")
line_name = os.path.expandvars("${HOME_TWOCRYST}/lossmaps_ALADDIN/opt_round_150_1500_optphases_b2.json")
input_file = Path(line_name)
#particle_ref_HL = xp.Particles(p0c=7000e9, q0=1, mass0=xp.PROTON_MASS_EV)

output_file = input_file.parent / (input_file.stem + '_patched.json')
print(output_file)

line = xt.Line.from_json(input_file)
print(f"Imported {len(line.element_names)} elements.")
#line.particle_ref = particle_ref_HL

# Elements to keep
collimators = [name for name in line.element_names
                    if (name.startswith('tc') or name.startswith('td'))
                    and not '_aper' in name and not name[-4:-2]=='mk' and not name[:4] == 'tcds'
                    and not name[:4] == 'tcdd' and not name[:5] == 'tclim' and not name[:3] == 'tca'
                    and not (name[-5]=='.' and name[-3]=='.') and not name[:5] == 'tcdqm'
              ]
# collimator_apertures = [f'{coll}_aper' + p for p in ['', '_patch'] for coll in collimators]
# ips = [f'ip{i+1}' for i in range(8)]


# Patch the aperture model by fixing missing apertures
def patch_aperture_with(missing, patch):
    if isinstance(missing, str) or not hasattr(missing, '__iter__'):
        missing = [missing]
    for nn in missing:
        if nn not in line.element_names:
            print(f"Element {nn} not found in line! Skipping aperture patching..")
            continue
        if isinstance(patch, str):
            if patch not in line.element_names:
                raise ValueError("Could not find patch aperture!")
            patch = line[patch].copy()
        line.insert_element(index=nn, element=patch,
                            name=nn+'_aper_patch')




if beam == 2:

    patch_aperture_with(['acfca.4al1.b2', 'acfca.4bl1.b2'], xt.LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315))
    patch_aperture_with(['acfca.4br5.b2', 'acfca.4ar5.b2'], xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a=0.04, b=0.04))
    patch_aperture_with(['acfca.4al5.b2', 'acfca.4bl5.b2'], xt.LimitRectEllipse(max_x=0.043, max_y=0.043, a=0.043, b=0.043))
    patch_aperture_with(['acfca.4br1.b2', 'acfca.4ar1.b2'], xt.LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315))

else:
    patch_aperture_with(['acfca.4ar1.b1', 'acfca.4br1.b1'], xt.LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315))
    patch_aperture_with(['acfca.4bl5.b1', 'acfca.4al5.b1'], xt. LimitRectEllipse(max_x=0.0414, max_y=0.0414, a=0.0414, b=0.0414))
    patch_aperture_with(['acfca.4ar5.b1', 'acfca.4br5.b1'], xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a=0.04, b=0.04))
    patch_aperture_with(['acfca.4bl1.b1', 'acfca.4al1.b1'], xt.LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315))




print("Aperture check after patching:")
#df_patched = line.check_aperture(needs_aperture=collimators)
df_patched = line.check_aperture()
assert not np.any(df_patched.has_aperture_problem)


# Save to json
with open(output_file, 'w') as fid:
    json.dump(line.to_dict(), fid, cls=xo.JEncoder, indent=True)


# Load from json to check that there are no loading errors
print("Reloading file to test json is not corrupted..")
with open(output_file, 'r') as fid:
    loaded_dct = json.load(fid)
newline = xt.Line.from_dict(loaded_dct)
# Temporary hack, as xt._lines_equal fails with compounds:
# apertures are a set, not a list, and between to_dict and from_dict the order is not kept
#line.compound_container = xt.compounds.CompoundContainer()
#newline.compound_container = xt.compounds.CompoundContainer()
assert xt._lines_equal(line, newline)
print("All done.")


"""
['acfca.4al1.b2',  LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315)
 'acfca.4bl1.b2',
 'acfca.4br5.b2', LimitRectEllipse(max_x=0.04, max_y=0.04, a=0.04, b=0.04)
 'acfca.4ar5.b2',
 'acfca.4al5.b2', LimitRectEllipse(max_x=0.043, max_y=0.043, a=0.043, b=0.043)
 'acfca.4bl5.b2',
 'acfca.4br1.b2',LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315)
 'acfca.4ar1.b2']
"""


"""
['acfca.4ar1.b1',  LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315)
 'acfca.4br1.b1',
 'acfca.4bl5.b1',  LimitRectEllipse(max_x=0.0414, max_y=0.0414, a=0.0414, b=0.0414)
 'acfca.4al5.b1',
 'acfca.4ar5.b1',  LimitRectEllipse(max_x=0.04, max_y=0.04, a=0.04, b=0.04)
 'acfca.4br5.b1',
 'acfca.4bl1.b1', LimitRectEllipse(max_x=0.0315, max_y=0.0315, a=0.0315, b=0.0315)
 'acfca.4al1.b1']
"""