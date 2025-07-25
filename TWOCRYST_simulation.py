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
from scipy.stats import chi

from IPython import embed

from xcoll.interaction_record.interaction_types import shortcuts




# ---------------------------- LOADING FUNCTIONS ----------------------------
def get_df_to_save(dict, df_part, num_particles, num_turns, df_imp = None, epsilon = 0, x_dim = None, y_dim = None, jaw_L = None, plane = 'V'):

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

    if plane == 'V':
        abs_y_low = jaw_L if jaw_L is not None else -0.04
        abs_y_up = jaw_L + y_dim if y_dim is not None else 0.04
        abs_x_low = -x_dim/2 if x_dim is not None else -0.04
        abs_x_up = x_dim/2 if x_dim is not None else 0.04
    elif plane == 'H':
        abs_x_low = jaw_L if jaw_L is not None else -0.04
        abs_x_up = jaw_L + x_dim if x_dim is not None else 0.04
        abs_y_low = -y_dim/2 if y_dim is not None else -0.04
        abs_y_up = y_dim/2 if y_dim is not None else 0.04
    
    for part in range(num_particles):
        for turn in range(num_turns):
            if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low - epsilon) and var_dict['x'][part, turn] < (abs_x_up + epsilon) and var_dict['y'][part, turn] > (abs_y_low - epsilon) and var_dict['y'][part, turn] < (abs_y_up + epsilon ):
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

    fixed_seed    = bool(run_dict['fixed_seed'])
    seed          = run_dict['seed']

    adt_amplitude    =  None if run_dict['adt_amplitude']== 'None' else float(run_dict['adt_amplitude'])

    if fixed_seed:
        np.random.seed(seed=seed)
        print('\n----- Seed set to: ', seed)

    TCCS_align_angle_step = float(run_dict['TCCS_align_angle_step'])
    TCCP_align_angle_step = float(run_dict['TCCP_align_angle_step'])
    TCCP_align_angle_additional = float(run_dict['TCCP_align_angle_additional']) if 'TCCP_align_angle_additional' in run_dict.keys() else 0

    TCCS_potential = None if run_dict['TCCS_potential'] == 'default' else float(run_dict['TCCS_potential'])
    TCCP_potential = None if run_dict['TCCP_potential'] == 'default' else float(run_dict['TCCP_potential'])

    normalized_emittance = run_dict['normalized_emittance']

    target_mode = run_dict['target_mode']
    input_mode = run_dict['input_mode']
    output_mode = run_dict['output_mode']
    turn_on_cavities = bool(run_dict['turn_on_cavities'])
    print('\nTarget mode: ', target_mode, '\t', 'input mode: ', input_mode, '\t', 'output mode: ', output_mode, '\t',  'Seed: ', seed, '\tCavities on: ', turn_on_cavities , '\t ADT: ', adt_amplitude, '\n')

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
    for gap_name in ['TCCS_gap', 'TCCP_gap', 'TARGET_gap', 'PIXEL_gap', 'TFT_gap', 'TCP_gap', 'TCLA_gap', 'TCT_gap']:
        if gap_name in run_dict.keys():
            gap = run_dict[gap_name]
            if gap == 'None':
                gaps[gap_name]= None
            elif gap == 'default':
                pass
            else:
                gaps[gap_name]= float(gap)  

    part_energy = None if run_dict['energy'] == 'None' else float(run_dict['energy'])

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
    if part_energy is not None:
        line.particle_ref = xt.Particles(p0c=part_energy, #eV
                                 q0=1, mass0=xt.PROTON_MASS_EV)
    print(f'\nParticle energy: {float(line.particle_ref.p0c)/1e9:} GeV\n')

    end_s = line.get_length()

    TCCS_name = 'tccs.5r3.b2'
    TCCP_name = 'tccp.4l3.b2'
    TARGET_name = 'target.4l3.b2'
    PIXEL_name = 'pixel.detector'
    TFT_name = 'tft.detector'
    TCP_name = 'tcp.d6r7.b2'
    TCLA_name = 'tcla.a5l3.b2'

    ydim_PIXEL = 0.01408
    xdim_PIXEL = 0.04224 #0.04246

    ydim_TFT = 0.029698484809835
    xdim_TFT = 0.04525483399593905

    RPX_bottom_wall_thickess = 2.14e-3
    PIX_y_distance_to_RPX = 2.94e-3

   
    
    TCLA_loc        = line.get_s_position()[line.element_names.index(TCLA_name)]
    TFT_loc         = end_s - 6651.9178 
    PIX2_loc        = end_s - 6652.61443 
    PIX2_loc_ff     = end_s - 6652.61453 
    PIX1_loc        = end_s - 6652.64117 
    PIX1_loc_ff     = end_s - 6652.64127  
    TCCP_loc        = end_s - 6653.2543
    TCCP_loc_ff     = end_s - 6653.2893 
    TARGET_loc      = end_s - 6653.29205
    TARGET_loc_ff   = end_s - 6653.29455 
    TCCS_loc        = end_s - 6773.9000
    TCCS_loc_ff     = end_s - 6773.9020
    TCP_loc         = line.get_s_position()[line.element_names.index(TCP_name)]
  

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
    #line.insert_element(at_s=PIX3_loc, element=xt.Marker(), name=PIXEL_name+'_3')
    #line.insert_element(at_s=PIX3_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name= PIXEL_name+'_3' + '_aper')
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


    # ---------------------------- SETUP ADT ----------------------------
    if adt_amplitude is not None:
        # Install ADT into line
        # ADT kickers in LHC are named adtk[hv].[abcd]5[lr]4.b1 (with the position 5l4 (B1H or B2V) or 5r4 (B1V or B2H)
        # These are not in the line, but their tank names are: adtk[hv].[abcd]5[lr]4.[abcd].b1  (32 markers)
        print(f"\nSetting up the ADT with aplitude {adt_amplitude}\n")
        pos = 'b5l4' if f'{beam}' == '1' and plane == 'H' else 'b5r4'
        pos = 'b5l4' if f'{beam}' == '2' and plane == 'V' else pos
        name = f'adtk{plane.lower()}.{pos}.b{beam}'
        tank_start = f'adtk{plane.lower()}.{pos}.a.b{beam}'
        tank_end   = f'adtk{plane.lower()}.{pos}.d.b{beam}'
        adt_pos = 0.5*line.get_s_position(tank_start) + 0.5*line.get_s_position(tank_end)
        adt = xc.BlowUp.install(line, name=f'{name}_blowup', at_s=adt_pos, plane=plane, stop_at_turn=num_turns,
                        amplitude=adt_amplitude, use_individual_kicks=True)
    


    # ---------------------------- SETUP OPTICS ----------------------------
    # Build the tracker
    line.build_tracker()
    line.collimators.assign_optics()



    # ---------------------------- SETUP COLLIMATORS ----------------------------
    # Set the collimator gaps
    print('\n---- Collimator gaps alignment ----')
    if 'TCCS_gap' in gaps.keys():
        line[TCCS_name].gap = gaps['TCCS_gap']
        print('TCCS gap set to: ', line[TCCS_name].gap)
    else:
        gaps['TCCS_gap'] = line[TCCS_name].gap
    if 'TCCP_gap' in gaps.keys():
        line[TCCP_name].gap = gaps['TCCP_gap']
        print('TCCP gap set to: ', line[TCCP_name].gap) 
    else:
        gaps['TCCP_gap'] = line[TCCP_name].gap
    if 'TARGET_gap' in gaps.keys():
        line[TARGET_name].gap = gaps['TARGET_gap']
        print('TARGET gap set to: ', line[TARGET_name].gap)
    else:
        gaps['TARGET_gap'] = line[TARGET_name].gap  
    if 'TCP_gap' in gaps.keys():
        line[TCP_name].gap = gaps['TCP_gap']
        print('TCP', TCP_name, 'gap set to: ', line[TCP_name].gap, ':', line[TCP_name].gap_L, line[TCP_name].gap_R)
    else:
        gaps['TCP_gap'] = line[TCP_name].gap_L
    if 'PIXEL_gap' in gaps.keys():
        #for pix_idx in ['_1', '_2', '_3']:
        for pix_idx in ['_1', '_2']:
            line[PIXEL_name + pix_idx].gap = gaps['PIXEL_gap']
            print(PIXEL_name + pix_idx, 'gap set to: ', line[PIXEL_name + pix_idx].gap)
    else:
        gaps['PIXEL_gap'] = line[PIXEL_name + '_1'].gap
    if 'TCLA_gap' in gaps.keys():
        line[TCLA_name].gap = gaps['TCLA_gap']
        print('TCLA gap set to: ', line[TCLA_name].gap)
    else:
        gaps['TCLA_gap'] = line[TCLA_name].gap

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
    print("TCCP align angle incremented by step: ", TCCP_align_angle_step, 'and additional: ', TCCP_align_angle_additional)
    line[TCCP_name].tilt = line[TCCP_name].tilt + TCCP_align_angle_step + TCCP_align_angle_additional
    print("TCCP final alignment angle: ", line[TCCP_name].tilt)

    if TCCS_potential is not None:
        print(f'\n---- changing TCCS potential to {TCCS_potential} ----')
        line[TCCS_name].material.crystal_potential = TCCS_potential
    if TCCP_potential is not None:
        print(f'\n---- changing TCCP potential to {TCCP_potential}----')
        line[TCCP_name].material.crystal_potential = TCCP_potential


    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)



    # ---------------------------- CALCULATE TWISS ----------------------------
    tw = line.twiss()
    print("Computed twiss.. \n")

    if adt_amplitude is not None:
        if plane == 'H':
            adt.calibrate_by_emittance(nemitt=normalized_emittance, twiss=tw)
        else:
            adt.calibrate_by_emittance(nemitt=normalized_emittance, twiss=tw)

    # ---------------------------- SETUP MONITORS ----------------------------
    line.discard_tracker()

    eps =  1e-8

    if 'TCCS_impacts' in save_list:
        TCCS_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        tilt_face_shift_TCCS = coll_dict[TCCS_name]["width"]*np.sin(line[TCCS_name].tilt) if tw['alfy', TCCS_name] < 0 else 0
        #line.insert_element(at_s = TCCS_loc - coll_dict[TCCS_name]["length"]/2 -tilt_face_shift_TCCS, element=TCCS_monitor, name='TCCS_monitor')
        line.insert_element(at_s = TCCS_loc_ff - tilt_face_shift_TCCS , element=TCCS_monitor, name='TCCS_monitor')
        print('... TCCS monitor inserted \n')

    if 'TARGET_impacts' in save_list:
        TARGET_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TARGET_loc_ff, element=TARGET_monitor, name='TARGET_monitor')
        print('... TARGET monitor inserted\n')

    if 'TCCP_impacts' in save_list:
        TCCP_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        tilt_face_shift_TCCP = coll_dict[TCCP_name]["width"]*np.sin(line[TCCP_name].tilt) if tw['alfy', TCCP_name] < 0 else 0
        #line.insert_element(at_s = TCCP_loc - coll_dict[TCCP_name]["length"]/2 - tilt_face_shift_TCCP, element=TCCP_monitor, name='TCCP_monitor')
        line.insert_element(at_s = TCCP_loc_ff - tilt_face_shift_TCCP - eps, element=TCCP_monitor, name='TCCP_monitor')
        print('... TCCP monitor inserted\n')

    if 'PIXEL_impacts' in save_list or 'PIXEL_impacts_1' in save_list or 'PIXEL_impacts_ALL' in save_list:
        PIXEL_monitor_1 = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        #embed()
        line.insert_element(at_s = PIX1_loc_ff , element=PIXEL_monitor_1, name='PIXEL_monitor_1')
        print('... PIXEL 1 monitor inserted\n')

    if 'PIXEL_impacts_2' in save_list or 'PIXEL_impacts_ALL' in save_list:
        PIXEL_monitor_2 = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = PIX2_loc_ff , element=PIXEL_monitor_2, name='PIXEL_monitor_2')
        print('... PIXEL 2 monitor inserted\n')

    #if 'PIXEL_impacts_3' in save_list or 'PIXEL_impacts_ALL' in save_list:
    #    PIXEL_monitor_3 = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
    #    line.insert_element(at_s = PIX3_loc - coll_dict[PIXEL_name+'_3']["length"]/2, element=PIXEL_monitor_3, name='PIXEL_impacts_3')
    #    print('\n... PIXEL 3 monitor inserted')

    if 'TFT_impacts' in save_list:
        TFT_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TFT_loc, element=TFT_monitor, name='TFT_monitor')
        print('... TFT monitor inserted\n')

    if 'TCLA_impacts' in save_list:
        TCLA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TCLA_loc - line[TCLA_name].length/2, element=TCLA_monitor, name='TCLA_monitor') 
        print('... TCLA monitor inserted\n')


    embed()
    # ---------------------------- CALCULATE INFO ----------------------------
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    # Printout useful informations
    idx_TCCS = line.element_names.index(TCCS_name)
    idx_TARGET = line.element_names.index(TARGET_name)
    idx_TCCP = line.element_names.index(TCCP_name)
    idx_PIXEL_1 = line.element_names.index(PIXEL_name + '_1')
    idx_PIXEL_2 = line.element_names.index(PIXEL_name + '_2')
    #idx_PIXEL_3 = line.element_names.index(PIXEL_name + '_3')
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



    # ---------------------------- SETUP IMPACTS ----------------------------
    print("\n... Setting up impacts\n")
    impacts = xc.InteractionRecord.start(line= line)  #capacity=int(2e7)

    # ---------------------------- INPUT GENERATION ----------------------------
    if input_mode == 'pencil_TCP':
        print("\n... Generating initial particles on TCP \n")
        # Generate initial pencil distribution on horizontal collimator
        impact_parameter = 0 
        part = line[TCP_name].generate_pencil(num_particles = num_particles, impact_parameter = impact_parameter)
        #part = xc.generate_pencil_on_collimator(name = TCP_name, line = line, num_particles=num_particles)
        part.at_element = idx_TCP 
        part.start_tracking_at_element = idx_TCP 

    elif input_mode == 'pencil_TCCS':
        print("\n... Generating initial particles on TCCS\n")
        #transverse_spread_sigma = 1
        #part = xc.generate_pencil_on_collimator(TCCS_name, num_particles=num_particles, transverse_spread_sigma=transverse_spread_sigma)
        part = line[TCCS_name].generate_pencil(num_particles = num_particles)
        #part = xc.generate_pencil_on_collimator(name = TCCS_name, line=line, num_particles=num_particles)
        if 'TCCS_monitor' in line.element_names:
            idx_monitor = line.element_names.index('TCCS_monitor')
        else:
            idx_monitor = line.element_names.index(TCCS_name)
        part.at_element = idx_monitor 
        part.start_tracking_at_element = idx_monitor
    
    elif input_mode == 'pencil_TCCP':
        print("\n... Generating initial particles on TCCP\n")
        #transverse_spread_sigma = 1
        #part = xc.generate_pencil_on_collimator(TCCS_name, num_particles=num_particles, transverse_spread_sigma=transverse_spread_sigma)
        part = line[TCCP_name].generate_pencil(num_particles = num_particles)
        #part = xc.generate_pencil_on_collimator(name = TCCS_name, line=line, num_particles=num_particles)
        if 'TCCP_monitor' in line.element_names:
            idx_monitor = line.element_names.index('TCCP_monitor')
        else:
            idx_monitor = line.element_names.index(TCCP_name)
        part.at_element = idx_monitor 
        part.start_tracking_at_element = idx_monitor

    elif input_mode == 'circular_halo':
        print("\n... Generating 2D uniform circular sector\n")
        if 'adt_limits' in run_dict.keys():
            adt_limits = run_dict['adt_limits']
        else:
            adt_limits = [0.003,0.002]
        ip1_idx = line.element_names.index('ip1')
        at_s = line.get_s_position(ip1_idx)
        gap = 5  
        print('adt_limits: ', adt_limits, 'gap: ', gap)
        # Vertical plane: generate cut halo distribution
        (y_in_sigmas, py_in_sigmas, r_points, theta_points
            )= xp.generate_2D_uniform_circular_sector(
                                                num_particles=num_particles,
                                                r_range=(gap - adt_limits[0], gap+adt_limits[1]), # sigmas  r_range=(gap - 0.003, gap+0.002)
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

    elif input_mode == 'gaussian_halo':
        chi_dist = chi(2)

        at_element = TCCS_name
        sigma_min = gaps['TCCS_gap'] - 0.003
        sigma_max = gaps['TCCS_gap'] + 0.002
        cdf_min = chi_dist.cdf(sigma_min)
        cdf_max = chi_dist.cdf(sigma_max)

        u = np.random.uniform(cdf_min, cdf_max, num_particles)
        r = chi_dist.ppf(u)
        theta = np.random.uniform(0, 2 * np.pi, num_particles)

        y_norm = r * np.cos(theta)
        py_norm = r * np.sin(theta)
        x_norm, px_norm = xp.generate_2D_gaussian(num_particles=num_particles)
        #zeta, delta = xp.generate_longitudinal_coordinates(num_particles=num_particles, distribution='gaussian', sigma_z=sigma_z, line=line)

        part = line.build_particles(
            #zeta=zeta,
            #delta=delta,
            x_norm=x_norm,
            px_norm=px_norm,
            y_norm=y_norm,
            py_norm=py_norm,
            nemitt_x=normalized_emittance,
            nemitt_y=normalized_emittance, 
            at_element=at_element
        )


    # ---------------------------- SEED FIXING ----------------------------
  
    if fixed_seed:
        print("\n... Fixing seed of particles\n")
        random_array = np.random.randint(0, 4291630464,  size = num_particles*4)
        part._rng_s1 = random_array[0:num_particles]
        part._rng_s2 = random_array[num_particles:num_particles*2]
        part._rng_s3 = random_array[num_particles*2:num_particles*3]
        part._rng_s4 = random_array[num_particles*3:num_particles*4]


    # ---------------------------- TRACKING ----------------------------
    #line.optimize_for_tracking()
    line.scattering.enable() 
    if adt_amplitude is not None:
        adt.activate()
  
    line.track(part, num_turns=num_turns, time=True)
    if adt_amplitude is not None:
        adt.deactivate()
    print("\nTCCS critical angle: ", line[idx_TCCS].critical_angle)
    print("TCCP critical angle: ", line[idx_TCCP].critical_angle)
    line.scattering.disable()
    impacts.stop()
    print(f"\nDone tracking in {line.time_last_track:.1f}s.")

    # Printout useful informations
    print("\n----- Check information -----")
    print(f"Line index of TCCS: {idx_TCCS}, TARGET: {idx_TARGET}, TCCP: {idx_TCCP}, PIXEL 1: {idx_PIXEL_1}, PIXEL 2: {idx_PIXEL_2}, TCP: {idx_TCP}\n")




    # ---------------------------- LOSSMAPS ----------------------------    
    if 'losses' in save_list:
        line.discard_tracker()
        line.build_tracker(_context=xo.ContextCpu())
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

    def calculate_xpcrit(name, line=line):
        bending_radius = line[name].bending_radius
        dp = 1.92e-10 
        pot_crit = 21.34
        eta = 0.9
        Rcrit = line.particle_ref.p0c/(2*np.sqrt(eta)*pot_crit) * (dp/2)
        xp_crit = np.sqrt(2*eta*pot_crit/line.particle_ref.p0c)*(1 - Rcrit/bending_radius)
        return xp_crit[0]


    tccs_imp = impacts.interactions_per_collimator(TCCS_name).reset_index()
    n_TCCS_abs = tccs_imp['int'].apply(lambda x: 'A'  in x).sum()
    print(f"\nTCCS: {n_TCCS_abs} particles absorbed")
    tccs_sim_chann_eff = None
    if len(tccs_imp) > 0:
        unique_values, counts = np.unique(tccs_imp['int'], return_counts=True)
        summary_int = pd.DataFrame({'int': unique_values,'counts': counts})
        summary_int.int = summary_int.int.astype(str)
        if "['CH']" in summary_int.int.to_list():
            tccs_sim_chann_eff = summary_int[summary_int['int'] == "['CH']"].counts.iloc[0] / sum((impacts.at_element == idx_TCCS) & (impacts.interaction_type == "Enter Jaw L") 
                                                                                             & (impacts.px_before  - miscut_TCCS < calculate_xpcrit(TCCS_name))&(impacts.px_before - miscut_TCCS > - calculate_xpcrit(TCCS_name)))
    print("TCCS channeling efficiency: ", tccs_sim_chann_eff, '\n')
    tccs_imp = tccs_imp.groupby('pid').agg(list).reset_index()[['pid', 'turn']]
    tccs_imp.rename(columns={'turn': 'TCCS_turn', 'pid':'particle_id'}, inplace=True)
    df_part = pd.merge(df_part, tccs_imp, on='particle_id', how='left')
    del tccs_imp
    gc.collect()
    
    tccp_imp = impacts.interactions_per_collimator(TCCP_name).reset_index()
    tccp_imp.rename(columns={'turn': 'this_turn', 'pid':'particle_id'}, inplace=True)
    m_entry =(impacts.at_element == idx_TCCP) & (impacts.interaction_type == "Enter Jaw L")
    tccp_imp = pd.merge(tccp_imp, pd.DataFrame({'this_turn':  impacts.at_turn[m_entry],'particle_id': impacts.id_before[m_entry],'py': impacts.px_before[m_entry]}),
                        on=['this_turn', 'particle_id'], how='left')
    n_TCCP_abs = tccp_imp['int'].apply(lambda x: 'A'  in x).sum()
    n_TCCP_abs_xp = tccp_imp[(tccp_imp.py - miscut_TCCP < calculate_xpcrit(TCCP_name))&(tccp_imp.py - miscut_TCCP > - calculate_xpcrit(TCCP_name))].int.apply(lambda x: 'A'  in x).sum()
    n_TCCP_abs_xp_2 =  tccp_imp[(tccp_imp.py - miscut_TCCP < calculate_xpcrit(TCCP_name)/2)&(tccp_imp.py - miscut_TCCP > - calculate_xpcrit(TCCP_name)/2)].int.apply(lambda x: 'A'  in x).sum()
    print(f"\nTCCP: {n_TCCP_abs} particles absorbed")
    tccp_sim_chann_eff = None
    if len(tccp_imp) > 0:
        if len(tccp_imp[tccp_imp.int.astype(str) == "['CH']"]) > 0:
            tccp_sim_chann_eff = len(tccp_imp[tccp_imp.int.astype(str) == "['CH']"]) /  len(tccp_imp[(tccp_imp["py"] - miscut_TCCP < calculate_xpcrit(TCCP_name)) &(tccp_imp["py"] - miscut_TCCP > -calculate_xpcrit(TCCP_name))])
    print("TCCP channeling efficiency: ", tccp_sim_chann_eff, '\n')
    
    tccp_imp = tccp_imp.groupby('particle_id').agg(list).reset_index()[['particle_id', 'this_turn']]
    tccp_imp.rename(columns={'this_turn': 'TCCP_turn'}, inplace=True)
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
    metadata = {'p0c': line.particle_ref.p0c[0], 'mass0': line.particle_ref.mass0, 'q0': line.particle_ref.q0, 'gamma0': line.particle_ref.gamma0[0], 'beta0': line.particle_ref.beta0[0], 
                'TCCS_absorbed': n_TCCS_abs, 'TCCP_absorbed': n_TCCP_abs, 
                'TCCP_absorbed_xp': n_TCCP_abs_xp, 'TCCP_absorbed_xp_2': n_TCCP_abs_xp_2,
                'TCCS_sim_chann_eff': tccs_sim_chann_eff, 'TCCP_sim_chann_eff': tccp_sim_chann_eff,
                'TCCP_jaw_U': line.elements[idx_TCCP].jaw_U, 'TCCP_jaw_D':line.elements[idx_TCCP].jaw_D,
                'TCCP_sigma': sigma_TCCP,
                'TCCS_gap': gaps['TCCS_gap'], 'TCCP_gap': gaps['TCCP_gap'], 'TARGET_gap': gaps['TARGET_gap'], 'PIXEL_gap': gaps['PIXEL_gap'], 'TFT_gap': gaps['TFT_gap'], 'TCP_gap': gaps['TCP_gap'],
                'TCCS_align_angle': line.elements[idx_TCCS].tilt, 'TCCP_align_angle': line.elements[idx_TCCP].tilt, 'TCCS_miscut': miscut_TCCS, 'TCCP_miscut': miscut_TCCP,}
    pd.DataFrame(list(metadata.values()), index=metadata.keys()).to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='metadata', format='table', mode='a',
            complevel=9, complib='blosc')


    if 'TCCS_impacts' in save_list:
        # SAVE IMPACTS ON TCCS
        print("... Saving impacts on TCCS\n(epsilon: ", epsilon_TCCS, ")\n")

        TCCS_monitor_dict = TCCS_monitor.to_dict()
        
        ydim_TCCS = coll_dict[TCCS_name]['width']
        xdim_TCCS =  coll_dict[TCCS_name]['height']
        jaw_L_TCCS = line.elements[idx_TCCS].jaw_U
        
        impact_part_df = get_df_to_save(TCCS_monitor_dict, df_part,  jaw_L = jaw_L_TCCS, #x_dim = xdim_TCCS, 
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
        print("... Saving impacts on TCCP\n(epsilon: ", epsilon_TCCP, ")\n")

        TCCP_monitor_dict = TCCP_monitor.to_dict()
        
        ydim_TCCP = coll_dict[TCCP_name]['width']
        xdim_TCCP =  coll_dict[TCCP_name]['height']
        jaw_L_TCCP = line.elements[idx_TCCP].jaw_U

        impact_part_df = get_df_to_save(TCCP_monitor_dict, df_part, jaw_L = jaw_L_TCCP,  # x_dim = xdim_TCCP, 
                epsilon = epsilon_TCCP,  num_particles=num_particles, num_turns=num_turns,
                df_imp = impacts.interactions_per_collimator(TCCP_name).reset_index())
        
        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCCP_impacts', format='table', mode='a',
            complevel=9, complib='blosc')

        del impact_part_df
        gc.collect()
        

    if 'TARGET_impacts' in save_list:

        # SAVE IMPACTS ON TARGET
        print("... Saving impacts on TARGET\n(epsilon: ", epsilon_TARGET, ")\n")

        TARGET_monitor_dict = TARGET_monitor.to_dict()
       
        ydim_TARGET = coll_dict[TARGET_name]['width']
        xdim_TARGET =  coll_dict[TARGET_name]['height']
        jaw_L_TARGET = line.elements[idx_TARGET].jaw_LU 

        impact_part_df = get_df_to_save(TARGET_monitor_dict, df_part, jaw_L = jaw_L_TARGET, # x_dim = xdim_TARGET, y_dim = ydim_TARGET,
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
        

    if 'PIXEL_impacts' in save_list or 'PIXEL_impacts_1' in save_list or 'PIXEL_impacts_ALL' in save_list:

        print("... Saving impacts on PIXEL 1\n(epsilon: ", epsilon_PIXEL, ")\n")


        if output_mode == 'packed':
            df_imp = impacts.interactions_per_collimator(TCCP_name).reset_index()
        elif output_mode == 'packed_TCCS':
            df_imp = impacts.interactions_per_collimator(TCCS_name).reset_index()
        else:
            df_imp = None

        PIXEL_monitor_dict = PIXEL_monitor_1.to_dict()
    
        jaw_L_PIXEL = sigma_PIXEL * gaps['PIXEL_gap'] + tw['y',PIXEL_name + '_1'] 

        impact_part_df = get_df_to_save(PIXEL_monitor_dict, df_part,  jaw_L = jaw_L_PIXEL,  #x_dim = xdim_PIXEL, y_dim = ydim_PIXEL,
                epsilon = epsilon_PIXEL, num_particles=num_particles, num_turns=num_turns, 
                df_imp = df_imp)

        if output_mode == 'packed':
            imp_TCCP_py  =  pd.DataFrame({'this_turn':  impacts.at_turn[m_entry],'particle_id': impacts.id_before[m_entry],'py': impacts.px_before[m_entry]})
            imp_TCCP_py['xp_crit']= np.where((imp_TCCP_py["py"] - miscut_TCCP < calculate_xpcrit(TCCP_name) / 2) &(imp_TCCP_py["py"] - miscut_TCCP > -calculate_xpcrit(TCCP_name) / 2), 2,
                np.where((imp_TCCP_py["py"] - miscut_TCCP < calculate_xpcrit(TCCP_name)) &(imp_TCCP_py["py"] - miscut_TCCP> -calculate_xpcrit(TCCP_name)),1,0)).astype('int32')
            impact_part_df = pd.merge(impact_part_df, imp_TCCP_py.drop(columns=['py']), on=['particle_id', 'this_turn'], how='left')
            impact_part_df = impact_part_df.drop(columns=['zeta', 'delta', 'TCP_turn'])

        if output_mode == 'packed_TCCS':
            m_entry_TCCS =(impacts.at_element == idx_TCCS) & (impacts.interaction_type == "Enter Jaw L")
            imp_TCCS_py  =  pd.DataFrame({'this_turn':  impacts.at_turn[m_entry_TCCS],'particle_id': impacts.id_before[m_entry_TCCS],'py': impacts.px_before[m_entry_TCCS]})
            imp_TCCS_py['xp_crit']= np.where((imp_TCCS_py["py"] - miscut_TCCS < calculate_xpcrit(TCCS_name) / 2) &(imp_TCCS_py["py"] - miscut_TCCS > -calculate_xpcrit(TCCS_name) / 2), 2,
                np.where((imp_TCCS_py["py"] - miscut_TCCS < calculate_xpcrit(TCCS_name)) &(imp_TCCS_py["py"] - miscut_TCCS> -calculate_xpcrit(TCCS_name)),1,0)).astype('int32')
            impact_part_df = pd.merge(impact_part_df, imp_TCCS_py.drop(columns=['py']), on=['particle_id', 'this_turn'], how='left')
            impact_part_df = impact_part_df.drop(columns=['zeta', 'delta', 'TCP_turn', 'TCCP_turn', 'at_element'])


        del PIXEL_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='PIXEL_impacts_1', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()





    if 'PIXEL_impacts_2' in save_list or 'PIXEL_impacts_ALL' in save_list:

        print("... Saving impacts on PIXEL 2\n(epsilon: ", epsilon_PIXEL, ")\n")

        PIXEL_monitor_dict = PIXEL_monitor_2.to_dict()
    
        jaw_L_PIXEL = sigma_PIXEL * gaps['PIXEL_gap'] + tw['y',PIXEL_name + '_2'] 

        impact_part_df = get_df_to_save(PIXEL_monitor_dict, df_part,  jaw_L = jaw_L_PIXEL,  #x_dim = xdim_PIXEL, y_dim = ydim_PIXEL,
                epsilon = epsilon_PIXEL, num_particles=num_particles, num_turns=num_turns,   
                df_imp = df_imp)
        
        del PIXEL_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='PIXEL_impacts_2', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()




    #if 'PIXEL_impacts_3' in save_list or 'PIXEL_impacts_ALL' in save_list:

    #    print("... Saving impacts on PIXEL 3\n(epsilon: ", epsilon_PIXEL, ")\n")

    #    PIXEL_monitor_dict = PIXEL_monitor_3.to_dict()

    #    jaw_L_PIXEL = sigma_PIXEL * gaps['PIXEL_gap'] + tw['y',PIXEL_name + '_3']

    #    impact_part_df = get_df_to_save(PIXEL_monitor_dict, df_part,  jaw_L = jaw_L_PIXEL,  #x_dim = xdim_PIXEL, y_dim = ydim_PIXEL,
    #            epsilon = epsilon_PIXEL, num_particles=num_particles, num_turns=num_turns,
    #            df_imp = df_imp)
        
    #    del PIXEL_monitor_dict
    #    gc.collect()

    #    if output_mode == 'reduced':
    #        impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

    #    impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='PIXEL_impacts_3', format='table', mode='a',
    #        complevel=9, complib='blosc')
        
    #    del impact_part_df
    #    gc.collect()




    if 'TFT_impacts' in save_list:


        if output_mode == 'packed':
            df_imp = impacts.interactions_per_collimator(TCCP_name).reset_index()
        elif output_mode == 'packed_TCCS':
            df_imp = impacts.interactions_per_collimator(TCCS_name).reset_index()
        else:
            df_imp = None

        # SAVE IMPACTS ON TFT
        print("... Saving impacts on TFT\n(epsilon: ", epsilon_TFT, ")\n")

        TFT_monitor_dict = TFT_monitor.to_dict()
    
        jaw_L_TFT = sigma_TFT * gaps['TFT_gap'] + tw['y',TFT_name] 

        impact_part_df = get_df_to_save(TFT_monitor_dict, df_part,  jaw_L = jaw_L_TFT,  #x_dim = xdim_TFT, y_dim = ydim_TFT,
                epsilon = epsilon_TFT, num_particles=num_particles, num_turns=num_turns, 
                df_imp = df_imp)
        
        del TFT_monitor_dict
        gc.collect()

        if output_mode == 'packed':
            imp_TCCP_py  =  pd.DataFrame({'this_turn':  impacts.at_turn[m_entry],'particle_id': impacts.id_before[m_entry],'py': impacts.px_before[m_entry]})
            imp_TCCP_py['xp_crit']= np.where((imp_TCCP_py["py"] - miscut_TCCP < calculate_xpcrit(TCCP_name) / 2) &(imp_TCCP_py["py"] - miscut_TCCP > -calculate_xpcrit(TCCP_name) / 2), 2,
                np.where((imp_TCCP_py["py"] - miscut_TCCP < calculate_xpcrit(TCCP_name)) &(imp_TCCP_py["py"] - miscut_TCCP> -calculate_xpcrit(TCCP_name)),1,0)).astype('int32')
            impact_part_df = pd.merge(impact_part_df, imp_TCCP_py.drop(columns=['py']), on=['particle_id', 'this_turn'], how='left')
            impact_part_df = impact_part_df.drop(columns=['zeta', 'delta', 'TCP_turn'])

        if output_mode == 'packed_TCCS':
            m_entry_TCCS =(impacts.at_element == idx_TCCS) & (impacts.interaction_type == "Enter Jaw L")
            imp_TCCS_py  =  pd.DataFrame({'this_turn':  impacts.at_turn[m_entry_TCCS],'particle_id': impacts.id_before[m_entry_TCCS],'py': impacts.px_before[m_entry_TCCS]})
            imp_TCCS_py['xp_crit']= np.where((imp_TCCS_py["py"] - miscut_TCCS < calculate_xpcrit(TCCS_name) / 2) &(imp_TCCS_py["py"] - miscut_TCCS > -calculate_xpcrit(TCCS_name) / 2), 2,
                np.where((imp_TCCS_py["py"] - miscut_TCCS < calculate_xpcrit(TCCS_name)) &(imp_TCCS_py["py"] - miscut_TCCS> -calculate_xpcrit(TCCS_name)),1,0)).astype('int32')
            impact_part_df = pd.merge(impact_part_df, imp_TCCS_py.drop(columns=['py']), on=['particle_id', 'this_turn'], how='left')
            impact_part_df = impact_part_df.drop(columns=['zeta', 'delta', 'TCP_turn', 'TCCP_turn', 'at_element'])

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TFT_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()

    if 'TCLA_impacts' in save_list:

        if output_mode == 'packed':
            df_imp = impacts.interactions_per_collimator(TCCP_name).reset_index()
        elif output_mode == 'packed_TCCS':
            df_imp = impacts.interactions_per_collimator(TCCS_name).reset_index()
        else:
            df_imp = None

        # SAVE IMPACTS ON TCLA
        print("... Saving impacts on TCLA\n, epsilon: ", epsilon_TCLA)

        TCLA_monitor_dict = TCLA_monitor.to_dict()
    
        jaw_L_TCLA = line.elements[idx_TCLA].jaw_LU

        impact_part_df = get_df_to_save(TCLA_monitor_dict, df_part,  jaw_L = jaw_L_TCLA,
                num_particles=num_particles, num_turns=num_turns, epsilon = epsilon_TCLA, 
                df_imp = df_imp)
        del TCLA_monitor
        gc.collect()

        if output_mode == 'packed_TCCS':
            m_entry_TCCS =(impacts.at_element == idx_TCCS) & (impacts.interaction_type == "Enter Jaw L")
            imp_TCCS_py  =  pd.DataFrame({'this_turn':  impacts.at_turn[m_entry_TCCS],'particle_id': impacts.id_before[m_entry_TCCS],'py': impacts.px_before[m_entry_TCCS]})
            imp_TCCS_py['xp_crit']= np.where((imp_TCCS_py["py"] - miscut_TCCS < calculate_xpcrit(TCCS_name) / 2) &(imp_TCCS_py["py"] - miscut_TCCS > -calculate_xpcrit(TCCS_name) / 2), 2,
                np.where((imp_TCCS_py["py"] - miscut_TCCS < calculate_xpcrit(TCCS_name)) &(imp_TCCS_py["py"] - miscut_TCCS> -calculate_xpcrit(TCCS_name)),1,0)).astype('int32')
            impact_part_df = pd.merge(impact_part_df, imp_TCCS_py.drop(columns=['py']), on=['particle_id', 'this_turn'], how='left')
            impact_part_df = impact_part_df.drop(columns=['zeta', 'delta', 'TCP_turn', 'TCCP_turn', 'at_element'])
        
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