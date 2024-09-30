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
import xfields as xf
import xdeps as xd
import scipy
import gc
import io 

from IPython import embed

from xcoll.interaction_record.interaction_types import shortcuts




# ---------------------------- LOADING FUNCTIONS ----------------------------
def get_df_to_save(dict, df_part, num_particles, num_turns, df_imp = None, epsilon = 0, start = False, x_dim = None, y_dim = None, jaw_L = None):

    float_variables = ['zeta', 'x', 'px', 'y', 'py', 'delta']
    int_variables = ['at_turn', 'particle_id', 'state']
    variables = float_variables + int_variables

    var_dict = {}

    for var in variables:
        var_dict[var] = dict['data'][var].reshape((num_particles, num_turns))   
    del dict
    gc.enable()
    gc.collect()

    
    impact_part_dict = {}
    for key in var_dict.keys():
        impact_part_dict[key] = []

    if x_dim is not None and jaw_L is not None and y_dim is not None:

        abs_y_low = jaw_L
        abs_y_up = jaw_L + y_dim
        abs_x_low = -x_dim/2
        abs_x_up = x_dim/2

        for part in range(num_particles):
            for turn in range(num_turns):
                if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low - epsilon) and var_dict['x'][part, turn] < (abs_x_up + epsilon) and var_dict['y'][part, turn] > (abs_y_low - epsilon) and var_dict['y'][part, turn] < (abs_y_up + epsilon):
                    for key in var_dict.keys():
                        impact_part_dict[key].append(var_dict[key][part, turn])

    elif x_dim is None and y_dim is None and jaw_L is not None:
        abs_y_low = jaw_L

        for part in range(num_particles):
            for turn in range(num_turns):
                if var_dict['state'][part, turn] > 0 and var_dict['y'][part, turn] > (abs_y_low - epsilon):
                    for key in var_dict.keys():
                        impact_part_dict[key].append(var_dict[key][part, turn])

    else:
        for part in range(num_particles):
            for turn in range(num_turns):
                if start and turn > 0:
                    continue
                else: 
                    if var_dict['state'][part, turn] > 0:
                        for key in var_dict.keys():
                            impact_part_dict[key].append(var_dict[key][part, turn])

    del var_dict
    gc.collect()


    impact_part_df = pd.DataFrame(impact_part_dict)
    del impact_part_dict
    gc.collect()
    
    
    impact_part_df.rename(columns={'at_turn': 'this_turn'}, inplace=True)
    impact_part_df = pd.merge(impact_part_df, df_part, on='particle_id', how='left')
    
    impact_part_df[float_variables] = impact_part_df[float_variables].astype('float32')
    impact_part_df[int_variables] = impact_part_df[int_variables].astype('int32')
    impact_part_df.drop('state', axis=1, inplace=True)
    impact_part_df['this_turn'] = impact_part_df['this_turn'].astype('int32')

    for col in ['TCCS_turn', 'TCCP_turn', 'TCP_turn']:
        impact_part_df[col] = impact_part_df[col].apply(lambda x: ','.join(map(str, x)))

    if df_imp is not None:
        df_imp.rename(columns={'turn': 'this_turn'}, inplace=True)
        df_imp.rename(columns={'pid': 'particle_id'}, inplace=True)
        df_imp.rename(columns={'int': 'interactions'}, inplace=True)
        df_imp['interactions'] = df_imp['interactions'].apply(lambda x: ','.join(map(str, x)))
        impact_part_df = pd.merge(impact_part_df, df_imp, on=['particle_id', 'this_turn'], how='left')
    return impact_part_df
    



# ---------------------------- MAIN ----------------------------


def main():

    print('\nxcoll version: ', xc.__version__)
    print('xtrack version: ', xt.__version__)
    print('xpart version: ', xp.__version__)
    print('xobjects version: ', xo.__version__)
    print('xfields version: ', xf.__version__)
    print('xdeps version: ', xd.__version__)


    config_file = sys.argv[1]
    
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # Configure run parameters
    run_dict = config_dict['run']

    beam          = run_dict['beam']
    plane         = run_dict['plane']

    num_turns     = run_dict['turns']
    num_particles = run_dict['nparticles']
    
    seed          = run_dict['seed']

    TCCS_align_angle_step = float(run_dict['TCCS_align_angle_step'])
    TCCP_align_angle_step = float(run_dict['TCCP_align_angle_step'])

    normalized_emittance = run_dict['normalized_emittance']

    target_mode = run_dict['target_mode']
    input_mode = run_dict['input_mode']
    output_mode = run_dict['output_mode']
    load_input_path = run_dict['load_input_path']
    turn_on_cavities = bool(run_dict['turn_on_cavities'])
    print('\nTarget mode: ', target_mode, '\t', 'input mode: ', input_mode, '\t', 'output mode: ', output_mode, '\t',  'Seed: ', seed, '\tCavities on: ', turn_on_cavities ,  '\n')

    save_list = run_dict['save_list']

    # Setup input files
    file_dict = config_dict['input_files']

    coll_file = os.path.expandvars(file_dict['collimators'])
    line_file = os.path.expandvars(file_dict[f'line_b{beam}'])
    
    print('\nInput files:\n', line_file, '\n', coll_file, '\n')

    if coll_file.endswith('.yaml'):
        with open(coll_file, 'r') as stream:
            coll_dict = yaml.safe_load(stream)['collimators']['b'+config_dict['run']['beam']]
    if coll_file.endswith('.data'):
        print("Please convert and use the yaml file for the collimator settings")
        sys.exit(1)
               
    gaps = {}
    for gap_name in ['TCCS_gap', 'TCCP_gap', 'TARGET_gap', 'PIXEL_gap', 'TFT_gap', 'TCP_gap']:
        gap = run_dict[gap_name]
        if gap == 'None':
            gaps[gap_name]= None
        else:
            gaps[gap_name]= float(gap)  

    epsilon_TCCS = float(run_dict['epsilon_TCCS'])
    epsilon_TCCP = float(run_dict['epsilon_TCCP'])
    epsilon_TARGET = float(run_dict['epsilon_TARGET'])
    epsilon_PIXEL = float(run_dict['epsilon_PIXEL'])
    epsilon_TFT = float(run_dict['epsilon_TFT'])
    epsilon_TCLA = float(run_dict['epsilon_TCLA'])

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
    TFT_name = 'tft.detector'
    TCP_name = 'tcp.d6r7.b2'
    TCLA_name = 'tcla.a5l3.b2'

    print('\n... PIXEL in the first roman pot\n')
    ydim_PIXEL = 0.01408
    xdim_PIXEL = 0.04224 #0.04246

    print('... TFT in the second roman pot\n')
    ydim_TFT = 0.029698484809835
    xdim_TFT = 0.04525483399593905

    RPX_bottom_wall_thickess = 2.14e-3
    PIX_y_distance_to_RPX = 4.26e-3

    TCCS_loc_abs  = 6773.9428  #6773.7 #6775
    TCCP_loc_abs  = 6653.2543  #6653.3 #6655
    PIX1_loc_abs = 6652.7039
    PIX2_loc_abs = 6652.6929
    PIX3_loc_abs = 6652.6819
    TFT_loc_abs = 6652.114

    TCCS_loc = end_s - TCCS_loc_abs
    TCCP_loc = end_s - TCCP_loc_abs
    TARGET_loc = end_s - (TCCP_loc_abs + coll_dict[TCCP_name]["length"]/2 + coll_dict[TARGET_name]["length"]/2)
    PIX1_loc = end_s - PIX1_loc_abs
    PIX2_loc = end_s - PIX2_loc_abs
    PIX3_loc = end_s - PIX3_loc_abs
    TFT_loc = end_s - TFT_loc_abs
    TCP_loc = line.get_s_position()[line.element_names.index(TCP_name)]
    TCLA_loc = line.get_s_position()[line.element_names.index(TCLA_name)]


    line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)
    line.insert_element(at_s=TCCS_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCS_name+'_aper')
    line.insert_element(at_s=TCCP_loc, element=xt.Marker(), name=TCCP_name)
    line.insert_element(at_s=TCCP_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCP_name+'_aper')
    line.insert_element(at_s=TARGET_loc, element=xt.Marker(), name=TARGET_name)
    line.insert_element(at_s=TARGET_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name= TARGET_name + '_aper')
    line.insert_element(at_s=PIX1_loc, element=xt.Marker(), name=PIXEL_name+'_1')
    line.insert_element(at_s=PIX1_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name= PIXEL_name+'_1' + '_aper')
    line.insert_element(at_s=PIX2_loc, element=xt.Marker(), name=PIXEL_name+'_2')
    line.insert_element(at_s=PIX2_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name= PIXEL_name+'_2' + '_aper')
    line.insert_element(at_s=PIX3_loc, element=xt.Marker(), name=PIXEL_name+'_3')
    line.insert_element(at_s=PIX3_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name= PIXEL_name+'_3' + '_aper')
    line.insert_element(at_s=TFT_loc, element=xt.Marker(), name=TFT_name)

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


    # Initialise collimator database
    colldb = xc.CollimatorDatabase.from_yaml(coll_file, beam=beam, ignore_crystals=False)

    coll_names = colldb.collimator_names

    if target_mode == 'target_absorber': 
        black_absorbers = [TARGET_name,]
    else: 
        black_absorbers = []

    everest_colls = [name for name in coll_names if name not in black_absorbers]

    colldb.install_everest_collimators(line = line, names=everest_colls,verbose=True)
    colldb.install_black_absorbers(line = line, names = black_absorbers, verbose=True)

    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)

    # ---------------------------- SETUP IMPACTS ----------------------------


    print("\n... Setting up impacts\n")
    impacts = xc.InteractionRecord.start(line= line)  #capacity=int(2e7)

    # Build the tracker
    line.build_tracker()
    xc.assign_optics_to_collimators(line=line)
    
    # Set the collimator gaps
    line[TCCS_name].gap = gaps['TCCS_gap']
    print('TCCS gap set to: ', line[TCCS_name].gap)
    line[TCCP_name].gap = gaps['TCCP_gap']
    print('TCCP gap set to: ', line[TCCP_name].gap) 
    line[TARGET_name].gap = gaps['TARGET_gap']
    print('TARGET gap set to: ', line[TARGET_name].gap)

    for tcp in ['tcp.d6r7.b2', 'tcp.c6r7.b2', 'tcp.b6r7.b2']:
        line[tcp].gap = gaps['TCP_gap']
        print(f'TCP {tcp} gap set to: ', line[tcp].gap, ':', line[tcp].gap_L, line[tcp].gap_R)


    print('\n---- Crystal alignment ----')
    if 'miscut' in  coll_dict[TCCS_name].keys():
        miscut_TCCS = coll_dict[TCCS_name]['miscut']
    else:
        miscut_TCCS = 0
    if 'miscut' in  coll_dict[TCCP_name].keys():
        miscut_TCCP = coll_dict[TCCP_name]['miscut']
    else:
        miscut_TCCP = 0

    # Align crystals
    if gaps['TCCS_gap'] is not None: 
        line[TCCS_name].align_to_beam_divergence()
    if gaps['TCCP_gap'] is not None:
        line[TCCP_name].align_to_beam_divergence()

    print("\nTCCS aligned to beam: ", line[TCCS_name].tilt)
    line[TCCS_name].tilt = line[TCCS_name].tilt - miscut_TCCS
    print("TCCS corrected by miscut: ", line[TCCS_name].tilt)
    print("TCCS align angle incremented by step: ", TCCS_align_angle_step)
    line[TCCS_name].tilt = line[TCCS_name].tilt + TCCS_align_angle_step
    print("TCCS final alignment angle: ", line[TCCS_name].tilt)
    
    print("\nTCCP aligned to beam: ", line[TCCP_name].tilt)
    line[TCCP_name].tilt = line[TCCP_name].tilt - miscut_TCCP
    print("TCCP corrected by miscut: ", line[TCCP_name].tilt)
    print("TCCP align angle incremented by step: ", TCCP_align_angle_step)
    line[TCCP_name].tilt = line[TCCP_name].tilt + TCCP_align_angle_step
    print("TCCP final alignment angle: ", line[TCCP_name].tilt)


    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)

    # ---------------------------- CALCULATE TWISS ----------------------------
    tw = line.twiss()

    line.discard_tracker()
    # ---------------------------- SETUP MONITORS ----------------------------
    if 'TCCS_impacts' in save_list:
        TCCS_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        tilt_face_shift_TCCS = coll_dict[TCCS_name]["width"]*np.sin(line[TCCS_name].tilt) if tw['alfy', TCCS_name] < 0 else 0
        line.insert_element(at_s = TCCS_loc - coll_dict[TCCS_name]["length"]/2 -tilt_face_shift_TCCS, element=TCCS_monitor, name='TCCS_monitor')
        print('\n... TCCS monitor inserted')

    if 'TARGET_impacts' in save_list:
        TARGET_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TARGET_loc - coll_dict[TARGET_name]["length"]/2, element=TARGET_monitor, name='TARGET_monitor')
        print('\n... TARGET monitor inserted')

    if 'TCCP_impacts' in save_list:
        TCCP_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        tilt_face_shift_TCCP = coll_dict[TCCP_name]["width"]*np.sin(line[TCCP_name].tilt) if tw['alfy', TCCP_name] < 0 else 0
        TCCP_monitor_s = TCCP_loc - coll_dict[TCCP_name]["length"]/2 - tilt_face_shift_TCCP
        line.insert_element(at_s = TCCP_monitor_s, element=TCCP_monitor, name='TCCP_monitor')
        print('\n... TCCP monitor inserted')

    if 'PIXEL_impacts' or 'PIXEL_impacts_1' or 'PIXEL_impacts_ALL'  in save_list:
        PIXEL_monitor_1 = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = PIX1_loc - coll_dict[PIXEL_name+'_1']["length"]/2, element=PIXEL_monitor_1, name='PIXEL_monitor_1')
        print('\n... PIXEL 1 monitor inserted')

    if 'PIXEL_impacts_2' or 'PIXEL_impacts_ALL'  in save_list:
        PIXEL_monitor_2 = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = PIX2_loc - coll_dict[PIXEL_name+'_2']["length"]/2, element=PIXEL_monitor_2, name='PIXEL_monitor_2')
        print('\n... PIXEL 3 monitor inserted')

    if 'PIXEL_impacts_3' or 'PIXEL_impacts_ALL'  in save_list:
        PIXEL_monitor_3 = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = PIX3_loc - coll_dict[PIXEL_name+'_3']["length"]/2, element=PIXEL_monitor_3, name='PIXEL_impacts_3')
        print('\n... PIXEL 3 monitor inserted')

    if 'TFT_impacts' in save_list:
        TFT_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TFT_loc, element=TFT_monitor, name='TFT_monitor')
        print('\n... TFT monitor inserted')

    if 'TCP_generated' in save_list:
        TCP_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TCP_loc + coll_dict[TCP_name]["length"]/2, element=TCP_monitor, name='TCP_monitor') 
        print('\n... TCP monitor inserted')

    if 'TCLA_impacts' in save_list:
        TCLA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TCLA_loc - coll_dict[TCLA_name]["length"]/2, element=TCLA_monitor, name='TCLA_monitor') 
        print('\n... TCLA monitor inserted')
    line.build_tracker()

    #embed()

    # Printout useful informations
    idx_TCCS = line.element_names.index(TCCS_name)
    idx_TARGET = line.element_names.index(TARGET_name)
    idx_TCCP = line.element_names.index(TCCP_name)
    idx_PIXEL_1 = line.element_names.index(PIXEL_name + '_1')
    idx_PIXEL_2 = line.element_names.index(PIXEL_name + '_2')
    idx_PIXEL_3 = line.element_names.index(PIXEL_name + '_3')
    idx_TFT = line.element_names.index(TFT_name)
    idx_TCP = line.element_names.index(TCP_name)
    idx_TCLA = line.element_names.index(TCLA_name)
    
    beta_rel = float(line.particle_ref.beta0)
    gamma = float(line.particle_ref.gamma0)
    emittance_phy = normalized_emittance/(beta_rel*gamma)

    sigma_TCCS = np.sqrt(emittance_phy*tw['bety',TCCS_name])
    sigma_TCCP = np.sqrt(emittance_phy*tw['bety',TCCP_name])
    sigma_TARGET = np.sqrt(emittance_phy* tw['bety',TARGET_name])
    sigma_PIXEL = np.sqrt(emittance_phy*tw['bety',PIXEL_name+'_1'])
    sigma_TFT = np.sqrt(emittance_phy*tw['bety',TFT_name])
    sigma_TCP = np.sqrt(emittance_phy*tw['bety',TCP_name])
    sigma_TCLA = np.sqrt(emittance_phy*tw['bety',TCLA_name])
    
    print(f"\nTCCS\nCrystalAnalysis(n_sigma={line.elements[idx_TCCS].gap}, length={ coll_dict[ TCCS_name]['length']}, ydim={ coll_dict[ TCCS_name]['width']}, xdim={ coll_dict[ TCCS_name]['height']}," + 
        f"bending_radius={coll_dict[ TCCS_name]['bending_radius']}, align_angle={ line.elements[idx_TCCS].tilt}, miscut = {miscut_TCCS}, sigma={sigma_TCCS}, jaw_L={line.elements[idx_TCCS].jaw_U })")
    print(f"TARGET\nTargetAnalysis(n_sigma={line.elements[idx_TARGET].gap}, target_type='target', length={ coll_dict[ TARGET_name]['length']}, ydim={ coll_dict[ TARGET_name]['width']}, xdim={ coll_dict[ TARGET_name]['height']},"+
        f"sigma={sigma_TARGET}, jaw_L={line.elements[idx_TARGET].jaw_LU })")
    print(f"TCCP\nCrystalAnalysis(n_sigma={line.elements[idx_TCCP].gap}, length={ coll_dict[ TCCP_name]['length']}, ydim={ coll_dict[ TCCP_name]['width']}, xdim={ coll_dict[ TCCP_name]['height']},"+ 
        f"bending_radius={ coll_dict[ TCCP_name]['bending_radius']}, align_angle={line.elements[idx_TCCP].tilt}, miscut = {miscut_TCCP}, sigma={sigma_TCCP}, jaw_L={line.elements[idx_TCCP].jaw_U })")
    print(f"TCP\nTargetAnalysis(n_sigma={line.elements[idx_TCP].gap}, target_type='collimator', length={coll_dict[ TCP_name]['length']}, ydim={0.025}, xdim={0.025},"+ 
        f"sigma={sigma_TCP}, jaw_L={line.elements[idx_TCP].jaw_LU })")
    print(f"TCLA\nTargetAnalysis(n_sigma={line.elements[idx_TCLA].gap}, target_type='collimator', length={coll_dict[ TCLA_name]['length']}, ydim={0.025}, xdim={0.025},"+ 
        f"sigma={sigma_TCLA},  jaw_L={line.elements[idx_TCLA].jaw_LU })")
    print(f"PIXEL\nTargetAnalysis(n_sigma={gaps['PIXEL_gap']}, target_type = 'pixel', ydim={ydim_PIXEL}, xdim={xdim_PIXEL},"+ 
        f"sigma={sigma_PIXEL})")
    print(f"TFT\nTargetAnalysis(n_sigma={gaps['TFT_gap']}, target_type='alfa', ydim={round(ydim_TFT, 5)}, xdim={round(xdim_TFT, 5)},"+ 
        f"sigma={sigma_TFT})\n")


    # ---------------------------- TRACKING ----------------------------
    if input_mode == 'pencil_TCP':
        print("\n... Generating initial particles\n")
        # Generate initial pencil distribution on horizontal collimator
        part = xc.generate_pencil_on_collimator(name = TCP_name, line = line, num_particles=num_particles)
        part.at_element = idx_TCP 
        part.start_tracking_at_element = idx_TCP 

    elif input_mode == 'pencil_TCCS':
        print("\n... Generating initial particles\n")
        #transverse_spread_sigma = 1
        #part = xc.generate_pencil_on_collimator(TCCS_name, num_particles=num_particles, transverse_spread_sigma=transverse_spread_sigma)
        part = xc.generate_pencil_on_collimator(name = TCCS_name, line=line, num_particles=num_particles)
        if 'TCCS_monitor' in line.element_names:
            idx_monitor = line.element_names.index('TCCS_monitor')
        else:
            idx_monitor = line.element_names.index(TCCS_name)
        part.at_element = idx_monitor 
        part.start_tracking_at_element = idx_monitor

    elif input_mode == 'circular_halo':
        print("\n... Generating 2D uniform circular sector\n")
        ip1_idx = line.element_names.index('ip1')
        at_s = line.get_s_position(ip1_idx)
        # Vertical plane: generate cut halo distribution
        (y_in_sigmas, py_in_sigmas, r_points, theta_points
            )= xp.generate_2D_uniform_circular_sector(
                                                num_particles=num_particles,
                                                r_range=(gaps[TCCS_name] - 0.003,  gaps[TCCS_name]+0.002), # sigmas
                                                )

        x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)
        #transverse_spread_sigma = 0.01
        #x_in_sigmas   = np.random.normal(loc=3.45e-7, scale=transverse_spread_sigma, size=num_particles)
        #px_in_sigmas = np.random.normal(scale=transverse_spread_sigma, size=num_particles)

        part = line.build_particles(
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas,
            nemitt_x=normalized_emittance, nemitt_y=normalized_emittance, match_at_s=at_s, at_element=ip1_idx)
        
        part.at_element = ip1_idx 
        part.start_tracking_at_element = ip1_idx

    elif input_mode == 'load':
        print("\n... Loading initial particles\n")
        n_job = seed - 1
        input_path  = Path(load_input_path, f'Job.{n_job}/Outputdata/particles_B{beam}{plane}.h5')
        print('Particles read from: ', input_path)
        df_part = pd.read_hdf(input_path, key='initial_particles')
        dct_part = df_part.to_dict(orient='list')
        size_vars = (
            (xo.Int64, '_capacity'),
            (xo.Int64, '_num_active_particles'),
            (xo.Int64, '_num_lost_particles'),
            (xo.Int64, 'start_tracking_at_element'),
        )
        scalar_vars = (
             (xo.Float64, 'q0'),
             (xo.Float64, 'mass0'),
             (xo.Float64, 't_sim'),
          )
        for tt, nn in scalar_vars + size_vars:
            if nn in dct_part.keys() and not np.isscalar(dct_part[nn]):
                 dct_part[nn] = dct_part[nn][0]
        part = xp.Particles.from_dict(dct_part, load_rng_state=True)

        del df_part, dct_part
        gc.collect()

        part.at_element = idx + 2 
        part.start_tracking_at_element = idx + 2

    save_inital_particles = False
    if save_inital_particles:
        print("\n... Saving initial particles\n")
        part.to_pandas().to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='initial_particles', format='table', mode='a',
            complevel=9, complib='blosc')

    

    # Track
    xc.enable_scattering(line)
    line.track(part, num_turns=num_turns, time=True)
    print("\nTCCS critical angle: ", line[idx_TCCS].critical_angle)
    print("TCCP critical angle: ", line[idx_TCCP].critical_angle)
    xc.disable_scattering(line)
    print(f"\nDone tracking in {line.time_last_track:.1f}s.")



    # Printout useful informations
    print("\n----- Check information -----")
    print(f"Line index of TCCS: {idx_TCCS}, TARGET: {idx_TARGET}, TCCP: {idx_TCCP}, PIXEL 1: {idx_PIXEL_1}, PIXEL 2: {idx_PIXEL_2}, PIXEL 3: {idx_PIXEL_3}, TCP: {idx_TCP}\n")


    # ---------------------------- LOSSMAPS ----------------------------    
    if 'losses' in save_list:
        line_is_reversed = True if f'{beam}' == '2' else False
        ThisLM = xc.LossMap(line, line_is_reversed=line_is_reversed, part=part)
        print(ThisLM.summary, '\n')
        ThisLM.to_json(file=Path(path_out, f'lossmap_B{beam}{plane}.json'))
        #ThisLM.save_summary(file=Path(path_out, f'coll_summary_B{beam}{plane}.out'))

    

    # ---------------------------- SAVE DATA ----------------------------

    
    df_part = part.to_pandas()
    drop_list = ['chi', 'charge_ratio', 'pdg_id', 'rvv', 'rpp', '_rng_s1', '_rng_s2', '_rng_s3', '_rng_s4', 'weight', 'ptau', 'q0','gamma0','beta0', 'mass0', 'start_tracking_at_element', 's', 'p0c', 'parent_particle_id', 'state']
    float_variables = ['zeta', 'x', 'px', 'y', 'py', 'delta']
    int_variables = ['at_turn', 'particle_id', 'at_element']
    df_part.drop(drop_list, axis=1, inplace=True)
    df_part[float_variables] = df_part[float_variables].astype('float32')
    df_part[int_variables] = df_part[int_variables].astype('int32')
    df_part = df_part[['at_element','at_turn', 'particle_id']]

    elements_idx = {}
    for idx in df_part['at_element']:
        if idx not in elements_idx:
            elements_idx[idx] = line.element_names[idx]
    elements_idx[0] = 'alive'

    if output_mode != 'reduced':
        print("\n... Saving elements index\n")
        pd.DataFrame(list(elements_idx.values()), index=elements_idx.keys()).to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='idx', format='table', mode='a',
                complevel=9, complib='blosc')

        del elements_idx
        gc.collect()



    tccs_imp = impacts.interactions_per_collimator(TCCS_name).reset_index()
    n_TCCS_abs = tccs_imp['int'].apply(lambda x: 'A'  in x).sum()
    print(f"\TCCS: {n_TCCS_abs} particles absorbed\n")
    tccs_imp = tccs_imp.groupby('pid').agg(list).reset_index()[['pid', 'turn']]
    tccs_imp.rename(columns={'turn': 'TCCS_turn', 'pid':'particle_id'}, inplace=True)
    df_part = pd.merge(df_part, tccs_imp, on='particle_id', how='left')
    del tccs_imp
    gc.collect()
    
    tccp_imp = impacts.interactions_per_collimator(TCCP_name).reset_index()
    n_TCCP_abs = tccp_imp['int'].apply(lambda x: 'A'  in x).sum()
    print(f"\TCCP: {n_TCCP_abs} particles absorbed\n")
    tccp_imp = tccp_imp.groupby('pid').agg(list).reset_index()[['pid', 'turn']]
    tccp_imp.rename(columns={'turn': 'TCCP_turn', 'pid':'particle_id'}, inplace=True)
    df_part = pd.merge(df_part, tccp_imp, on='particle_id', how='left')
    del tccp_imp
    gc.collect()

    tcp_hor_imp = impacts.interactions_per_collimator('tcp.d6r7.b2').reset_index().groupby('pid').agg(list).reset_index()[['pid', 'turn']]
    tcp_ver_imp = impacts.interactions_per_collimator('tcp.c6r7.b2').reset_index().groupby('pid').agg(list).reset_index()[['pid', 'turn']]
    tcp_skw_imp = impacts.interactions_per_collimator('tcp.b6r7.b2').reset_index().groupby('pid').agg(list).reset_index()[['pid', 'turn']]
    merged_df = tcp_hor_imp.merge(tcp_ver_imp, on='pid', how='outer', suffixes=('_1', '_2')).merge(tcp_skw_imp, on='pid', how='outer')
    del tcp_hor_imp, tcp_ver_imp, tcp_skw_imp
    gc.collect()

    for column in ['turn_1', 'turn_2', 'turn']:
        merged_df[column] = merged_df[column].apply(lambda x: x if isinstance(x, list) else [])
    merged_df['turn_combined'] = merged_df.apply(lambda row: [row['turn_1'], row['turn_2'], row['turn']], axis=1)
    merged_df = merged_df.drop(columns=['turn_1', 'turn_2', 'turn'])
    merged_df.rename(columns={'turn_combined': 'TCP_turn', 'pid':'particle_id'}, inplace=True)

    df_part = pd.merge(df_part, merged_df,  on='particle_id', how='left')
    del merged_df
    gc.collect()
    
    for column in ['TCCS_turn', 'TCCP_turn', 'TCP_turn']:
        df_part[column] = df_part[column].apply(lambda x: x if isinstance(x, list) else [None])


    print("... Saving metadata\n")
    metadata = {'p0c': line.particle_ref.p0c[0], 'mass0': line.particle_ref.mass0, 'q0': line.particle_ref.q0, 'gamma0': line.particle_ref.gamma0[0], 'beta0': line.particle_ref.beta0[0], 'TCCS_absorbed': n_TCCS_abs, 'TCCP_absorbed': n_TCCP_abs}
    pd.DataFrame(list(metadata.values()), index=metadata.keys()).to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='metadata', format='table', mode='a',
            complevel=9, complib='blosc')

    if 'TCP_generated' in save_list:
        
        print("\n... Saving particles generated in interactions with TCP \n")

        TCP_monitor_dict = TCP_monitor.to_dict()

        impact_part_df = get_df_to_save(TCP_monitor_dict, df_part,
                                        num_particles=num_particles, num_turns=num_turns, start = True)
        
        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCP_generated', format='table', mode='a',
            complevel=9, complib='blosc')



    if 'TCCS_impacts' in save_list:
        # SAVE IMPACTS ON TCCS
        print("... Saving impacts on TCCS\n")

        TCCS_monitor_dict = TCCS_monitor.to_dict()
        
        ydim_TCCS = coll_dict[TCCS_name]['width']
        xdim_TCCS =  coll_dict[TCCS_name]['height']
        jaw_L_TCCS = line.elements[idx_TCCS].jaw_U
        
        impact_part_df = get_df_to_save(TCCS_monitor_dict, df_part, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = jaw_L_TCCS, 
                epsilon = epsilon_TCCS, num_particles=num_particles, num_turns=num_turns, 
                df_imp = impacts.interactions_per_collimator(TCCS_name).reset_index())
        
        del TCCS_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCCS_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()


    if 'TCCP_impacts' in save_list:
        # SAVE IMPACTS ON TCCP
        print("... Saving impacts on TCCP\n")

        TCCP_monitor_dict = TCCP_monitor.to_dict()
        
        ydim_TCCP = coll_dict[TCCP_name]['width']
        xdim_TCCP =  coll_dict[TCCP_name]['height']
        jaw_L_TCCP = line.elements[idx_TCCP].jaw_U
        
        impact_part_df = get_df_to_save(TCCP_monitor_dict, df_part, x_dim = xdim_TCCP, y_dim = ydim_TCCP, jaw_L = jaw_L_TCCP, 
                epsilon = epsilon_TCCP, num_particles=num_particles, num_turns=num_turns,
                df_imp = impacts.interactions_per_collimator(TCCP_name).reset_index())
        
        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCCP_impacts', format='table', mode='a',
            complevel=9, complib='blosc')

        del impact_part_df
        gc.collect()
        

    if 'TARGET_impacts' in save_list:

        # SAVE IMPACTS ON TARGET
        print("... Saving impacts on TARGET\n")

        TARGET_monitor_dict = TARGET_monitor.to_dict()
       
        ydim_TARGET = coll_dict[TARGET_name]['width']
        xdim_TARGET =  coll_dict[TARGET_name]['height']
        jaw_L_TARGET = line.elements[idx_TARGET].jaw_LU 

        impact_part_df = get_df_to_save(TARGET_monitor_dict, df_part, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = jaw_L_TARGET,
                epsilon = epsilon_TARGET, num_particles=num_particles, num_turns=num_turns,
                df_imp = impacts.interactions_per_collimator(TARGET_name).reset_index())
        
        del TARGET_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TARGET_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()
        

    if 'PIXEL_impacts' or 'PIXEL_impacts_1' or 'PIXEL_impacts_ALL' in save_list:

        print("... Saving impacts on PIXEL 1\n")

        PIXEL_monitor_dict = PIXEL_monitor_1.to_dict()
    
        jaw_L_PIXEL = sigma_PIXEL * gaps['PIXEL_gap'] + tw['y',PIXEL_name + '_1'] + PIX_y_distance_to_RPX     

        impact_part_df = get_df_to_save(PIXEL_monitor_dict, df_part,  jaw_L = jaw_L_PIXEL,  #x_dim = xdim_PIXEL, y_dim = ydim_PIXEL,
                epsilon = epsilon_PIXEL, num_particles=num_particles, num_turns=num_turns)
        
        del PIXEL_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='PIXEL_impacts_1', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()




    if 'PIXEL_impacts_2' or 'PIXEL_impacts_ALL' in save_list:

        print("... Saving impacts on PIXEL 2\n")

        PIXEL_monitor_dict = PIXEL_monitor_2.to_dict()
    
        jaw_L_PIXEL = sigma_PIXEL * gaps['PIXEL_gap'] + tw['y',PIXEL_name + '_2']  + PIX_y_distance_to_RPX    

        impact_part_df = get_df_to_save(PIXEL_monitor_dict, df_part,  jaw_L = jaw_L_PIXEL,  #x_dim = xdim_PIXEL, y_dim = ydim_PIXEL,
                epsilon = epsilon_PIXEL, num_particles=num_particles, num_turns=num_turns)
        
        del PIXEL_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='PIXEL_impacts_2', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()




    if 'PIXEL_impacts_3' or 'PIXEL_impacts_ALL' in save_list:

        print("... Saving impacts on PIXEL 3\n")

        PIXEL_monitor_dict = PIXEL_monitor_3.to_dict()
    
        jaw_L_PIXEL = sigma_PIXEL * gaps['PIXEL_gap'] + tw['y',PIXEL_name + '_2'] + PIX_y_distance_to_RPX

        impact_part_df = get_df_to_save(PIXEL_monitor_dict, df_part,  jaw_L = jaw_L_PIXEL,  #x_dim = xdim_PIXEL, y_dim = ydim_PIXEL,
                epsilon = epsilon_PIXEL, num_particles=num_particles, num_turns=num_turns)
        
        del PIXEL_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='PIXEL_impacts_3', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()




    if 'TFT_impacts' in save_list:

        # SAVE IMPACTS ON TFT
        print("... Saving impacts on TFT\n")

        TFT_monitor_dict = TFT_monitor.to_dict()
    
        jaw_L_TFT = sigma_TFT * gaps['TFT_gap'] + tw['y',TFT_name] + RPX_bottom_wall_thickess         

        impact_part_df = get_df_to_save(TFT_monitor_dict, df_part,  jaw_L = jaw_L_TFT,  #x_dim = xdim_TFT, y_dim = ydim_TFT,
                epsilon = epsilon_TFT, num_particles=num_particles, num_turns=num_turns)
        
        del TFT_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TFT_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()

    if 'TCLA_impacts' in save_list:

        # SAVE IMPACTS ON TCLA
        print("... Saving impacts on TCLA\n")

        TCLA_monitor_dict = TCLA_monitor.to_dict()
    
        jaw_L_TCLA = line.elements[idx_TCLA].jaw_LU

        impact_part_df = get_df_to_save(TCLA_monitor_dict, df_part,  jaw_L = jaw_L_TCLA,
                num_particles=num_particles, num_turns=num_turns, epsilon = epsilon_TCLA)
        
        del TCLA_monitor
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCLA_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()



if __name__ == "__main__":
    main()







# XT_LOST_ON_APERTURE       0
# XT_LOST_ON_LONG_CUT      -2
# XT_LOST_ALL_E_IN_SYNRAD -10
# RNG_ERR_SEEDS_NOT_SET   -20
# RNG_ERR_INVALID_TRACK   -21
# RNG_ERR_RUTH_NOT_SET    -22
# XC_LOST_ON_EVEREST_BLOCK   -330
# XC_LOST_ON_EVEREST_COLL    -331
# XC_LOST_ON_EVEREST_CRYSTAL -332
# XC_LOST_ON_FLUKA_BLOCK     -333
# XC_LOST_ON_FLUKA           -334
# XC_LOST_ON_FLUKA_CRYSTAL   -335
# XC_LOST_ON_GEANT4_BLOCK    -336
# XC_LOST_ON_GEANT4          -337
# XC_LOST_ON_GEANT4_CRYSTAL  -338
# XC_LOST_ON_ABSORBER        -340
# XC_ERR_INVALID_TRACK       -390
# XC_ERR_NOT_IMPLEMENTED     -391
# XC_ERR_INVALID_XOFIELD     -392
# XC_ERR                     -399