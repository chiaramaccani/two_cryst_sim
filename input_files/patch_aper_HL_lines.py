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
beam        = 2  # int(sys.argv[2])
line_name = os.path.expandvars("${HOME_TWOCRYST}/input_files/HL_IR7_IR3_rematched/b4_sequence.json")
input_file = Path(line_name)
particle_ref_HL = xp.Particles(p0c=7000e9, q0=1, mass0=xp.PROTON_MASS_EV)

output_file = input_file.parent / (input_file.stem + '_patched.json')
print(output_file)

line = xt.Line.from_json(input_file)
print(f"Imported {len(line.element_names)} elements.")
line.particle_ref = particle_ref_HL

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



patch_aperture_with(['tdisa.a4r8.b2', 'tdisb.a4r8.b2', 'tdisc.a4r8.b2'], 'vmbgd.4r8.d.b2_aper')
patch_aperture_with('tcld.a11l2.b2', 'lepra.11l2.b2_mkex_aper')
#patch_aperture_with('tcld.a11l2.b2', xt.LimitEllipse(a=4e-2, b=4e-2))
patch_aperture_with(['tcspm.b4r7.b2', 'tcspm.e5l7.b2', 'tcspm.6l7.b2'], 'tcspm.4r7.b.b2_aper')
patch_aperture_with('tcld.9l7.b2', 'mbh.a9l7.b2_mken_aper')



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
line.compound_container = xt.compounds.CompoundContainer()
newline.compound_container = xt.compounds.CompoundContainer()
assert xt._lines_equal(line, newline)
print("All done.")