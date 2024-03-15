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



def calcAction(p0,tw,ele,exn=2.5e-6,nrj=7000e9,debug=False):
    ex = exn*0.938e9/nrj
    alfx = tw.rows[ele].cols['alfx'].alfx[0]
    alfy = tw.rows[ele].cols['alfy'].alfy[0]
    betx = tw.rows[ele].cols['betx'].betx[0]
    bety = tw.rows[ele].cols['bety'].bety[0]
    xx0 = tw.rows[ele].cols['x'].x[0]
    yy0 = tw.rows[ele].cols['y'].y[0]
    pxx0 = tw.rows[ele].cols['px'].px[0]
    pyy0 = tw.rows[ele].cols['py'].py[0]
    x0 = p0.x - xx0
    y0 = p0.y - yy0
    px0 = p0.px - pxx0
    py0 = p0.py - pyy0
    jx = np.sqrt(x0**2/betx + (alfx*x0/np.sqrt(betx) + np.sqrt(betx)*px0)**2)/np.sqrt(ex)
    jy = np.sqrt(y0**2/bety + (alfy*y0/np.sqrt(bety) + np.sqrt(bety)*py0)**2)/np.sqrt(ex)

    if debug:
        print(tw.rows[ele].cols[['alfx','alfy','betx','bety','x','y','px','py']])
    return jx,jy

def plotHist(p1,p2,figdim=(20,10), nbins=200, save=False, savePath="plot", line = None, density=False,scale=1, range=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,figsize=figdim)
    ax.hist([p1*scale,p2*scale],nbins,density=density,color=['red','green'])
    if line is not None:
        if not hasattr(line, '__iter__') or isinstance(line, str):
            line = [line]
        for l in line:
            ax.axvline(x=l, color='black', linestyle='--')

    if range is not None:
        ax.set_xlim(range)
    plt.show()

def get_df_to_save(dict, num_particles, num_turns, epsilon = 0, start = False, x_dim = None, y_dim = None, jaw_L = None):

    float_variables = ['zeta', 'x', 'px', 'y', 'py', 'delta', 'p0c']
    int_variables = ['at_turn', 'particle_id', 'at_element', 'state', 'parent_particle_id']
    variables = float_variables + int_variables
    variables.remove('at_element')

    df = pd.DataFrame(dict['data'])
    var_dict = {}

    for var in variables:
        new_arr = np.array(df[var])
        new_arr = new_arr.reshape((num_particles, num_turns))
        var_dict[var] = new_arr   
    del df

    
    impact_part_dict = {}
    for key in var_dict.keys():
        impact_part_dict[key] = []

    if x_dim is not None and jaw_L is not None and y_dim is not None:

        abs_y_low = jaw_L
        abs_y_up = jaw_L + y_dim
        abs_x_low = -x_dim/2
        abs_x_up = x_dim/2

        print('x_dim', x_dim, 'y_dim', y_dim, 'jaw_L', jaw_L, num_particles, num_turns)
        for part in range(num_particles):
            for turn in range(num_turns):
                if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low - epsilon) and var_dict['x'][part, turn] < (abs_x_up + epsilon) and var_dict['y'][part, turn] > (abs_y_low - epsilon) and var_dict['y'][part, turn] < (abs_y_up + epsilon):
                    for key in var_dict.keys():
                        impact_part_dict[key].append(var_dict[key][part, turn])

    impact_part_df = pd.DataFrame(impact_part_dict) 
    
    return impact_part_df


line0_file = '${HOME_TWOCRYST}/input_files/HL_IR7_tune_changed/b4_sequence_patched_tune.json'
line1_file = '${HOME_TWOCRYST}/input_files/HL_IR7_phase_advance/b4_sequence_patched_phadv.json'


# -------------------------------------------------------------------------------

#import TWOCRYST_analysis as twa

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




beam          = 2
plane         = 'V'

num_turns     = 1 #run_dict['turns'] 
num_particles = 1000000 #len(start_values)

normalized_emittance = 2.5e-6


TCCS_name = 'tccs.5r3.b2'
TCCP_name = 'tccp.4l3.b2'
TARGET_name = 'target.4l3.b2'
PIXEL_name = 'pixel.detector'
TCP_name = 'tcp.d6r7.b2'

TCCS_align_angle_step = 0



coll_file = os.path.expandvars('${HOME_TWOCRYST}/input_files/colldbs/HL_tight_twocryst.yaml')




if coll_file.endswith('.yaml'):
    with open(coll_file, 'r') as stream:
        coll_dict = yaml.safe_load(stream)['collimators']['b2']
if coll_file.endswith('.data'):
    print("Please convert and use the yaml file for the collimator settings")
    sys.exit(1)
    


context = xo.ContextCpu(omp_num_threads='auto')



# ---------------------------- SETUP LINE ----------------------------
def setup_line(line_name, coll_dict=coll_dict, beam = 2, plane = 'V', num_particles = num_particles, num_turns = 1, TCCS_align_angle_step=0):

    TCCS_align_angle_step = 0
    TCCS_gap =  7.2 #7.2
    TARGET_gap = 33.6 #33.6
    TCCP_gap =  33.6 #33.6
    PIXEL_gap = 33.6  #33.6

    mode = 'target_absorber'
    turn_on_cavities = True
    engine        = 'everest'


    # Load from json
    line_file = os.path.expandvars(line_name)
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

    coll_manager.build_tracker()
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

    return line, coll_manager


"""part = xp.Particles(p0c = start_values['p0c'].values[0], x = start_values['x'].values, y = start_values['y'].values, 
                    px = start_values['px'].values, py = start_values['py'].values, 
                    zeta = start_values['zeta'].values, delta = start_values['delta'].values)
p0c = 7e12
x =  0.0001
y = 0.0001
px =  0.0001
py = 0.0001
zeta = 0.0001
delta = 0.0001

part = xp.Particles(p0c = p0c, x = x, y = y, 
                    px = px, py = py, 
                    zeta = zeta, delta = delta)

"""      


line0, coll_manager0 = setup_line(line0_file)
line1, coll_manager1 = setup_line(line1_file)

tcp  = f"tcp.{'c' if plane=='H' else 'd'}6{'l' if beam=='1' else 'r'}7.b{beam}"
part = coll_manager0.generate_pencil_on_collimator(tcp, num_particles=1e6, impact_parameter=2e-6)

#embed()





idx_TCP = line0.element_names.index(tcp)
idx_TARGET = line0.element_names.index(TARGET_name)

tw0 = line0.twiss()
tw1 = line1.twiss()

coll_manager0.enable_scattering()
coll_manager1.enable_scattering()

part_before_TCP = part.copy()
#line.element_dict['tcp.d6r7.b2'].track(part)


line0.track(part, num_turns=1, ele_stop='ip6')
jxT,jyT = calcAction(part,tw0,'tcp.d6r7.b2')
m1 = jyT>7.2
part_ip6 = part.filter(m1)
part_TCCS_10 = part_ip6.copy() # for tracking without rematched phase
part_TCCS_11 = part_ip6.copy() # for tracking with rematched phase
line0.track(part_TCCS_10, num_turns=1, ele_start='ip6', ele_stop=TCCS_name)
line1.track(part_TCCS_11, num_turns=1, ele_start='ip6', ele_stop=TCCS_name)

idx_TCCS = line0.element_names.index(TCCS_name)
ele2 = line0.element_names[idx_TCCS+2]

part_afterTCCS_20 = part_TCCS_10.copy()
part_afterTCCS_21 = part_TCCS_11.copy()
line0.track(part_afterTCCS_20, num_turns=1, ele_start=TCCS_name, ele_stop=ele2)
line1.track(part_afterTCCS_21, num_turns=1, ele_start=TCCS_name, ele_stop=ele2)
part_TARGET_30 = part_afterTCCS_20.copy()
part_TARGET_31 = part_afterTCCS_21.copy()
line0.track(part_TARGET_30, num_turns=1, ele_start=ele2, ele_stop=TARGET_name)
line1.track(part_TARGET_31, num_turns=1, ele_start=ele2, ele_stop=TARGET_name)
#part40 = part10.copy()
#line0.track(part40, num_turns=1, ele_start='ip6', ele_stop='ip5')

jaw_L_TARGET = line0.elements[idx_TARGET].jaw_L 
jaw_L_TCCS = line0.elements[idx_TCCS].jaw_L 

ydim_TCCS = coll_dict[TCCS_name]['xdim']
xdim_TCCS =  coll_dict[TCCS_name]['ydim']

       
ydim_TARGET = coll_dict[TARGET_name]['xdim']
xdim_TARGET =  coll_dict[TARGET_name]['ydim']    


"""hit_TCCS_0 = part_TCCS_10.filter((part_TCCS_10.y > line0.elements[idx_TCCS].jaw_L ) & (part_TCCS_10.y < line0.elements[idx_TCCS].jaw_L + ydim_TCCS) & (part_TCCS_10.x > -xdim_TCCS/2) & (part_TCCS_10.x < xdim_TCCS/2) & (part_TCCS_10.state == 1) )
hit_TCCS_1 = part_TCCS_11.filter((part_TCCS_11.y > line1.elements[idx_TCCS].jaw_L) & (part_TCCS_11.y < line1.elements[idx_TCCS].jaw_L + ydim_TCCS) & (part_TCCS_11.x > -xdim_TCCS/2) & (part_TCCS_11.x < xdim_TCCS/2) & (part_TCCS_11.state == 1) )
print(len(hit_TCCS_0.y))
print(len(hit_TCCS_1.y))


hit_TARGET_0 = part_TARGET_30.filter((part_TARGET_30.y > line0.elements[idx_TARGET].jaw_L ) & (part_TARGET_30.y < line0.elements[idx_TARGET].jaw_L + ydim_TARGET) & (part_TARGET_30.x > -xdim_TARGET/2) & (part_TARGET_30.x < xdim_TARGET/2) & (part_TARGET_30.state == 1) )
hit_TARGET_1 = part_TARGET_31.filter((part_TARGET_31.y > line1.elements[idx_TARGET].jaw_L) & (part_TARGET_31.y < line1.elements[idx_TARGET].jaw_L + ydim_TARGET) & (part_TARGET_31.x > -xdim_TARGET/2) & (part_TARGET_31.x < xdim_TARGET/2) & (part_TARGET_31.state == 1) )
print(len(hit_TARGET_0.y))
print(len(hit_TARGET_1.y))"""



hit_TCCS_0 = part_TCCS_10.filter((part_TCCS_10.y > (line0.elements[idx_TCCS].jaw_L + line0.elements[idx_TCCS].ref_y)) & (part_TCCS_10.y < (line0.elements[idx_TCCS].jaw_L + line0.elements[idx_TCCS].ref_y + ydim_TCCS)) & (part_TCCS_10.x > -xdim_TCCS/2) & (part_TCCS_10.x < xdim_TCCS/2) & (part_TCCS_10.state == 1))
hit_TCCS_1 = part_TCCS_11.filter((part_TCCS_11.y > (line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y)) & (part_TCCS_11.y < (line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y + ydim_TCCS)) & (part_TCCS_11.x > -xdim_TCCS/2) & (part_TCCS_11.x < xdim_TCCS/2) & (part_TCCS_11.state == 1))
print(len(hit_TCCS_0.y))
print(len(hit_TCCS_1.y))


hit_TARGET_0 = part_TARGET_30.filter((part_TARGET_30.y > (line0.elements[idx_TARGET].jaw_L + line0.elements[idx_TARGET].ref_y)) & (part_TARGET_30.y < (line0.elements[idx_TARGET].jaw_L + line0.elements[idx_TARGET].ref_y + ydim_TARGET)) & (part_TARGET_30.x > -xdim_TARGET/2) & (part_TARGET_30.x < xdim_TARGET/2) & (part_TARGET_30.state == 1) )
hit_TARGET_1 = part_TARGET_31.filter((part_TARGET_31.y > (line1.elements[idx_TARGET].jaw_L + line1.elements[idx_TARGET].ref_y)) & (part_TARGET_31.y < (line1.elements[idx_TARGET].jaw_L + line1.elements[idx_TARGET].ref_y + ydim_TARGET)) & (part_TARGET_31.x > -xdim_TARGET/2) & (part_TARGET_31.x < xdim_TARGET/2) & (part_TARGET_31.state == 1) )
print(len(hit_TARGET_0.y))
print(len(hit_TARGET_1.y))






jx0,jy0 = calcAction(part_before_TCP,tw0,'tcp.d6r7.b2')
jx1,jy1 = calcAction(part_ip6,tw0,'ip6')
jx2,jy2 = calcAction(part_TCCS_10,tw0,TCCS_name)
jx3,jy3 = calcAction(part_afterTCCS_20,tw0,ele2)
jx4,jy4 = calcAction(part_TARGET_30,tw0,TARGET_name)

print(np.mean(jy0))
print(np.mean(jy1[:-np.sum(part_ip6.state<0)]))
print(np.mean(jy2[:-np.sum(part_TCCS_10.state<0)]))
print(np.mean(jy3[:-np.sum(part_afterTCCS_20.state<0)]))
print(np.mean(jy4[:-np.sum(part_TARGET_30.state<0)]))




TARGET_monitor0 = line0.element_dict['TARGET_monitor']
TARGET_monitor_dict0 = TARGET_monitor0.to_dict()

TCCS_monitor0 = line0.element_dict['TCCS_monitor']
TCCS_monitor_dict0 = TCCS_monitor0.to_dict()
        
"""TARGET_imp_0 = get_df_to_save(TARGET_monitor_dict0, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line0.elements[idx_TARGET].jaw_L ,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_0 = get_df_to_save(TCCS_monitor_dict0, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line0.elements[idx_TCCS].jaw_L , 
                epsilon = 0, num_particles=num_particles, num_turns=num_turns)"""

TARGET_imp_0 = get_df_to_save(TARGET_monitor_dict0, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line0.elements[idx_TARGET].jaw_L + line0.elements[idx_TARGET].ref_y,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_0 = get_df_to_save(TCCS_monitor_dict0, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line0.elements[idx_TCCS].jaw_L + line0.elements[idx_TCCS].ref_y, 
                epsilon = 0, num_particles=num_particles, num_turns=num_turns)




TARGET_monitor1 = line1.element_dict['TARGET_monitor']
TARGET_monitor_dict1 = TARGET_monitor1.to_dict()

TCCS_monitor1 = line1.element_dict['TCCS_monitor']
TCCS_monitor_dict1 = TCCS_monitor1.to_dict()
        
TARGET_imp_1 = get_df_to_save(TARGET_monitor_dict1, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line1.elements[idx_TARGET].jaw_L + line1.elements[idx_TARGET].ref_y,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_1 = get_df_to_save(TCCS_monitor_dict1, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y, 
                epsilon = 0, num_particles=num_particles, num_turns=1)

"""TARGET_imp_1 = get_df_to_save(TARGET_monitor_dict1, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line1.elements[idx_TARGET].jaw_L ,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_1 = get_df_to_save(TCCS_monitor_dict1, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line1.elements[idx_TCCS].jaw_L , 
                epsilon = 0, num_particles=num_particles, num_turns=1)"""

print(len(hit_TCCS_0.y) == len(TCCS_imp_0.y))
print(len(hit_TCCS_1.y) == len(TCCS_imp_1.y))
print(len(hit_TARGET_0.y) == len(TARGET_imp_0.y))
print(len(hit_TARGET_1.y) == len(TARGET_imp_1.y))


embed()