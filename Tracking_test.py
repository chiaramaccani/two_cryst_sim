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


# ---------------------------- LOADING FUNCTIONS ----------------------------

def find_axis_intercepts(x_coords, y_coords):
    x_intercepts = []
    y_intercepts = []

    for i in range(len(x_coords)):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[(i + 1) % len(x_coords)], y_coords[(i + 1) % len(y_coords)]

        if x1 == x2:
        # Vertical line, no y-intercept
            y_intercept = 0.0 if x1 == x2 == 0.0 else None
        else:
            slope = (y2 - y1) / (x2 - x1)
            y_intercept = y1 - (slope * x1)

        if y1 == y2:
        # Horizontal line, no x-intercept
            x_intercept = 0.0 if y1 == y2 == 0.0 else None
        else:
            slope = (x2 - x1) / (y2 - y1)
            x_intercept = x1 - (slope * y1)

        # Check if the x-intercept is within the range of x1 and x2
        if x_intercept is not None and (x1 <= x_intercept <= x2 or x2 <= x_intercept <= x1):
            x_intercepts.append(x_intercept)

        # Check if the y-intercept is within the range of y1 and y2
        if y_intercept is not None and (y1 <= y_intercept <= y2 or y2 <= y_intercept <= y1):
            y_intercepts.append(y_intercept)

    return x_intercepts, y_intercepts



def find_bad_offset_apertures(line):
    aperture_offsets = {}
    for name, element in line.element_dict.items():
        if 'offset' in name and element.__class__.__name__.startswith('XYShift'):
            aper_name = name.split('_offset')[0]
            aperture_offsets[aper_name] = (element.dx, element.dy)

    bad_apers = {}
    for ap_name, offset in aperture_offsets.items():
        aperture_el = line.element_dict[ap_name]

        cname= aperture_el.__class__.__name__
        ap_dict = aperture_el.to_dict()

        if cname == 'LimitEllipse':
            x_min = -ap_dict['a']
            x_max = ap_dict['a']
            y_min = -ap_dict['b']
            y_max = ap_dict['b']
        elif cname == 'LimitRect':
            x_min = ap_dict['min_x']
            x_max = ap_dict['max_x']
            y_min = ap_dict['min_y']
            y_max = ap_dict['max_y']
        elif cname == 'LimitRectEllipse':
            x_min = -ap_dict['max_x']
            x_max = ap_dict['max_x']
            y_min = -ap_dict['max_y']
            y_max = ap_dict['max_y']
        elif cname == 'LimitRacetrack':
            x_min = ap_dict['min_x']
            x_max = ap_dict['max_x']
            y_min = ap_dict['min_y']
            y_max = ap_dict['max_y']
        elif cname == 'LimitPolygon':
            x_intercepts, y_intercepts = find_axis_intercepts(ap_dict['x_vertices'],
                                                            ap_dict['y_vertices'])
            x_min = min(x_intercepts)
            x_max = max(x_intercepts)
            y_min = min(y_intercepts)
            y_max = max(y_intercepts)

        tolerance = 5e-3
        """if (x_max - offset[0] < tolerance 
            or -x_min + offset[0] < tolerance 
            or y_max - offset[1] < tolerance 
            or -y_min + offset[1] < tolerance):"""
        if (offset[0] -x_max > tolerance 
            or  -offset[0] + x_min > tolerance 
            or  offset[1] - y_max > tolerance 
            or  -offset[1] + y_min > tolerance ):
                bad_apers[ap_name] = (x_min, x_max, y_min, y_max, offset[0], offset[1])

    return bad_apers



# -------------------------------------------------------------------------------

import TWOCRYST_analysis as twa

"""load_particles = True
df_key = "TCP_generated"    
file = 'TCP_generated.h5'

default_path = "/eos/home-c/cmaccani/xsuite_sim/two_cryst_sim/Condor/"
folder = 'TEST_prova__target_absorber_20240305-1759/'

if load_particles:
    TCP = twa.TargetAnalysis(n_sigma=5.916079783099615, length=0.6, ydim=0.025, xdim=0.025, sigma=0.00015657954897267004)
    TCP.load_particles(folder, df_key=df_key)   
    TCP.save_particles_data(file_name = file, output_path = default_path + folder,  df_key=df_key)
    start_values = TCP.data
else:
    start_values = pd.read_hdf(default_path + folder + file, key=df_key)"""


p0c = 7e12
x =  0.0001
y = 0.0001
px =  0.0001
py = 0.0001
zeta = 0.0001
delta = 0.0001


config_file = os.path.expandvars('$HOME_TWOCRYST/config_sim.yaml')

with open(config_file, 'r') as stream:
    config_dict = yaml.safe_load(stream)

# Configure run parameters
run_dict = config_dict['run']

beam          = run_dict['beam']
plane         = run_dict['plane']

num_turns     = 1 #run_dict['turns'] 
num_particles = 1#len(start_values)
engine        = run_dict['engine']

seed          = run_dict['seed']

TCCS_align_angle_step = float(run_dict['TCCS_align_angle_step'])

normalized_emittance = run_dict['normalized_emittance']

mode = run_dict['mode']
turn_on_cavities = bool(run_dict['turn_on_cavities'])
print('\nMode: ', mode, '\t', 'Seed: ', seed, '\tCavities on: ', turn_on_cavities ,  '\n')

save_list = run_dict['save_list']

# Setup input files
file_dict = config_dict['input_files']

coll_file = os.path.expandvars(file_dict['collimators'])
line_file = os.path.expandvars(file_dict[f'line_b{beam}'])

print('Input files:\n', line_file, '\n', coll_file, '\n')

if coll_file.endswith('.yaml'):
    with open(coll_file, 'r') as stream:
        coll_dict = yaml.safe_load(stream)['collimators']['b'+config_dict['run']['beam']]
if coll_file.endswith('.data'):
    print("Please convert and use the yaml file for the collimator settings")
    sys.exit(1)
    
    

TCCS_gap = float(run_dict['TCCS_gap'])
TCCP_gap = float(run_dict['TCCP_gap'])
TARGET_gap = float(run_dict['TARGET_gap'])
PIXEL_gap = float(run_dict['PIXEL_gap'])

context = xo.ContextCpu(omp_num_threads='auto')

# Define output path
path_out = Path.cwd() / 'Outputdata'

if not path_out.exists():
    os.makedirs(path_out)



# ---------------------------- SETUP LINE ----------------------------

# Load from json
line = xt.Line.from_json(line_file)

end_s = line.get_length()

TCCS_name = 'tccs.5r3.b2'
TCCP_name = 'tccp.4l3.b2'
TARGET_name = 'target.4l3.b2'
PIXEL_name = 'pixel.detector'
TCP_name = 'tcp.d6r7.b2'

d_pix = 1 # [m]
ydim_PIXEL = 0.01408
xdim_PIXEL = 0.04246

TCCS_loc = end_s - 6773.7 #6775
TCCP_loc = end_s - 6653.3 #6655
dx = 1e-11
TARGET_loc = end_s - (6653.3 + coll_dict[TCCP_name]["length"]/2 + coll_dict[TARGET_name]["length"]/2 + dx)
PIXEL_loc = end_s - (6653.3 - coll_dict[TCCP_name]["length"]/2 - d_pix)
TCP_loc = line.get_s_position()[line.element_names.index(TCP_name)]


line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)
line.insert_element(at_s=TCCS_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCS_name+'_aper')
line.insert_element(at_s=TCCP_loc, element=xt.Marker(), name=TCCP_name)
line.insert_element(at_s=TCCP_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCP_name+'_aper')
line.insert_element(at_s=TARGET_loc, element=xt.Marker(), name=TARGET_name)
line.insert_element(at_s=TARGET_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name= TARGET_name + '_aper')
line.insert_element(at_s=PIXEL_loc, element=xt.Marker(), name=PIXEL_name)


TCCS_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
TARGET_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
TCCP_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
PIXEL_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
TCP_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
#dx = 1e-11
line.insert_element(at_s = TCCS_loc - coll_dict[TCCS_name]["length"]/2 - dx, element=TCCS_monitor, name='TCCS_monitor')
line.insert_element(at_s = TARGET_loc - coll_dict[TARGET_name]["length"]/2 - dx, element=TARGET_monitor, name='TARGET_monitor')
line.insert_element(at_s = TCCP_loc - coll_dict[TCCP_name]["length"]/2 - dx/2, element=TCCP_monitor, name='TCCP_monitor')
line.insert_element(at_s = PIXEL_loc, element=PIXEL_monitor, name='PIXEL_monitor')
line.insert_element(at_s = TCP_loc + coll_dict[TCP_name]["length"]/2 + 1e5*dx, element=TCP_monitor, name='TCP_monitor') 

bad_aper = find_bad_offset_apertures(line)
print('Bad apertures : ', bad_aper)
print('Replace bad apertures with Marker')
for name in bad_aper.keys():
    line.element_dict[name] = xt.Marker()
    print(name, line.get_s_position(name), line.element_dict[name])

# switch on cavities
if turn_on_cavities:
    speed = line.particle_ref._xobject.beta0[0]*scipy.constants.c
    harmonic_number = 35640
    voltage = 12e6/len(line.get_elements_of_type(xt.Cavity)[1])
    frequency = harmonic_number * speed /line.get_length()
    for side in ['l', 'r']:
        for cell in ['a','b','c','d']:
            line[f'acsca.{cell}5{side}4.b2'].voltage = voltage
            line[f'acsca.{cell}5{side}4.b2'].frequency = frequency   

# Aperture model check
print('\nAperture model check on imported model:')
df_imported = line.check_aperture()
assert not np.any(df_imported.has_aperture_problem)


# Initialise collmanager
coll_manager = xc.CollimatorManager.from_yaml(coll_file, line=line, beam=beam, _context=context, ignore_crystals=False)

# Install collimators into line
if engine == 'everest':
    coll_names = coll_manager.collimator_names

    if mode == 'target_absorber': 
        black_absorbers = [TARGET_name,]
    else: 
        black_absorbers = []

    everest_colls = [name for name in coll_names if name not in black_absorbers]

    coll_manager.install_everest_collimators(names=everest_colls,verbose=True)
    coll_manager.install_black_absorbers(names = black_absorbers, verbose=True)
else:
    raise ValueError(f"Unknown scattering engine {engine}!")


# Aperture model check
print('\nAperture model check after introducing collimators:')
df_with_coll = line.check_aperture()
assert not np.any(df_with_coll.has_aperture_problem)

    
# Build the tracker
#coll_manager.build_tracker()
coll_manager.build_tracker()

# Set the collimator openings based on the colldb,
# or manually override with the option gaps={collname: gap}
#coll_manager.set_openings()
coll_manager.set_openings(gaps = {TCCS_name: TCCS_gap, TCCP_name: TCCP_gap, TARGET_name: TARGET_gap})


print("\nTCCS aligned to beam: ", line[TCCS_name].align_angle)
#line[TTCS_name].align_angle = TTCS_align_angle_step
print("TCCS align angle incremented by step: ", TCCS_align_angle_step)
line[TCCS_name].align_angle = line[TCCS_name].align_angle + TCCS_align_angle_step
print("TCCS final alignment angle: ", line[TCCS_name].align_angle)


# Aperture model check
print('\nAperture model check after introducing collimators:')
df_with_coll = line.check_aperture()
assert not np.any(df_with_coll.has_aperture_problem)


"""part = xp.Particles(p0c = start_values['p0c'].values[0], x = start_values['x'].values, y = start_values['y'].values, 
                    px = start_values['px'].values, py = start_values['py'].values, 
                    zeta = start_values['zeta'].values, delta = start_values['delta'].values)"""

part = xp.Particles(p0c = p0c, x = x, y = y, 
                    px = px, py = py, 
                    zeta = zeta, delta = delta)

idx = line.element_names.index(TCP_monitor)
part.at_element = idx
part.start_tracking_at_element = idx

coll_manager.enable_scattering()
line.track(part, num_turns=num_turns, time=True)
coll_manager.disable_scattering()

embed()