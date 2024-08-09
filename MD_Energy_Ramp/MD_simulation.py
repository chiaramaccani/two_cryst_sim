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
#import psutil

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

        """abs_y_low = jaw_L
        abs_y_up = jaw_L + y_dim
        abs_x_low = -x_dim/2
        abs_x_up = x_dim/2"""

        abs_y_low = -y_dim/2
        abs_y_up = y_dim/2
        abs_x_low = jaw_L
        abs_x_up = jaw_L + x_dim

        for part in range(num_particles):
            for turn in range(num_turns):
                #if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low - epsilon) and var_dict['x'][part, turn] < (abs_x_up + epsilon) and var_dict['y'][part, turn] > (abs_y_low - epsilon) and var_dict['y'][part, turn] < (abs_y_up + epsilon):
                if var_dict['state'][part, turn] > 0:    
                    for key in var_dict.keys():
                        impact_part_dict[key].append(var_dict[key][part, turn])

    elif x_dim is None and y_dim is None and jaw_L is not None:
        abs_x_low = jaw_L

        for part in range(num_particles):
            for turn in range(num_turns):
                if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low - epsilon):
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

    impact_part_df['CRY_turn'] = impact_part_df['CRY_turn'].apply(lambda x: ','.join(map(str, x)))

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

    normalized_emittance = 3.5e-6 
    energy = f'{run_dict["energy"]}'

    input_mode = run_dict['input_mode']
    output_mode = run_dict['output_mode']
    run_mode = run_dict['run_mode']     
    save_list = run_dict['save_list']



    # Setup input files
    file_dict = config_dict['input_files']

    coll_file = os.path.expandvars(file_dict['collimators'])
    line_file = os.path.expandvars(file_dict[f'line_b{beam}'])
    sim_dict = os.path.expandvars(file_dict['sim_dict'])
    
    print('\nInput files:\n', line_file, '\n', coll_file, '\n')

    with open(coll_file, 'r') as stream:
        coll_dict = yaml.safe_load(stream)['collimators']['b'+config_dict['run']['beam']]

    with open(sim_dict, 'r') as f:
        data = json.load(f)

    part_energy = data[energy]['energy'] 
    gaps = data[energy]['gap']


    epsilon_CRY = float(run_dict['epsilon_CRY'])
    epsilon_LIN = float(run_dict['epsilon_LIN'])
    CRY_align_angle_step = float(run_dict['CRY_align_angle_step'])


    context = xo.ContextCpu(omp_num_threads='auto')

    # Define output path
    path_out = Path.cwd() / 'Outputdata'

    if not path_out.exists():
        os.makedirs(path_out)



    # ---------------------------- SETUP LINE ----------------------------

    # Load from json
    line = xt.Line.from_json(line_file)
    line.particle_ref = xt.Particles(p0c=part_energy, #eV
                                 q0=1, mass0=xt.PROTON_MASS_EV)


    print("\nProton energy: ", part_energy)
    print('Mode: ', run_mode, '\n')

    end_s = line.get_length()

    CRY_name ='tcpch.a5r7.b2'
    TCP_name = 'tcp.d6r7.b2'
    LIN_name = 'tcsg.b4r7.b2'

    dx = 0
    CRY_loc = line.get_s_position()[line.element_names.index(CRY_name)]
    TCP_loc = line.get_s_position()[line.element_names.index(TCP_name)]
    LIN_loc = line.get_s_position()[line.element_names.index(LIN_name)]

    # Aperture model check
    print('\nAperture model check on imported model:')
    df_imported = line.check_aperture()
    assert not np.any(df_imported.has_aperture_problem)


    # Initialise collimator database
    colldb = xc.CollimatorDatabase.from_yaml(coll_file, beam=beam, ignore_crystals=False)

    coll_names = colldb.collimator_names
    black_absorbers = []
    everest_colls = [name for name in coll_names if name not in black_absorbers]

    colldb.install_everest_collimators(line = line, names=everest_colls,verbose=True)
    colldb.install_black_absorbers(line = line, names = black_absorbers, verbose=True)



    if 'CRY_impacts' in save_list:
        CRY_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = CRY_loc - coll_dict[CRY_name]["length"]/2 - dx, element=CRY_monitor, name='CRY_monitor') 
        print('\n... CRY monitor inserted')

    if 'LIN_SCAN_impacts' in save_list:
        LIN_SCAN_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = LIN_loc - coll_dict[LIN_name]["length"]/2 - dx, element=LIN_SCAN_monitor, name='LIN_monitor') 
        print('\n... TCSG (LIN SCAN) monitor inserted')

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

    print('\n---- Collimators alignment ----\n')
    if  run_mode == 'linear_scan': 
        LIN_gap = run_dict['LIN_gap']
        if LIN_gap == 'None':
            gaps[LIN_name]= None
        elif LIN_gap == 'closed':
            gaps[LIN_name]=  gaps[CRY_name]
        else:
            gaps[LIN_name]= float(LIN_gap)  
        
    # Set the collimator gaps
    for name, gap in gaps.items():
        if gap is not None:
            line[name].gap = gap
            print(f'{name} gap set to: ', line[name].gap)

    print('\n---- Crystal alignment ----')
    if 'miscut' in  coll_dict[CRY_name].keys():
        miscut = coll_dict[CRY_name]['miscut']
    else:
        miscut = 0
    # Align crystals
    line[CRY_name].align_to_beam_divergence()
    print("CRY aligned to beam: ", line[CRY_name].tilt)
    line[CRY_name].tilt = line[CRY_name].tilt - miscut
    print("CRY corrected by miscut: ", line[CRY_name].tilt)
    print("CRY align angle incremented by step: ", CRY_align_angle_step)
    line[CRY_name].tilt = line[CRY_name].tilt + CRY_align_angle_step
    print("CRY final alignment angle: ", line[CRY_name].tilt)


    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)
    
    # Printout useful informations
    idx_CRY = line.element_names.index(CRY_name)
    idx_TCP = line.element_names.index(TCP_name)
    idx_LIN = line.element_names.index(LIN_name)

    tw = line.twiss()
    beta_rel = float(line.particle_ref.beta0)
    gamma = float(line.particle_ref.gamma0)
    emittance_phy = normalized_emittance/(beta_rel*gamma)

    sigma_CRY = np.sqrt(emittance_phy*tw['betx',CRY_name])
    sigma_TCP = np.sqrt(emittance_phy*tw['betx',TCP_name])
    sigma_LIN = np.sqrt(emittance_phy*tw['betx',LIN_name])

    print(f"\nCRY\nCrystalAnalysis(plane='H', n_sigma={line.elements[idx_CRY].gap}, length={ coll_dict[ CRY_name]['length']}, ydim={ coll_dict[ CRY_name]['height']}, xdim={ coll_dict[ CRY_name]['width']}," + 
        f"bending_radius={coll_dict[ CRY_name]['bending_radius']}, align_angle={ line.elements[idx_CRY].tilt}, miscut = {miscut},sigma={sigma_CRY}, jaw_L={line.elements[idx_CRY].jaw_U })")
    print(f"LIN_SCAN\nTargetAnalysis(plane='H', n_sigma={line.elements[idx_LIN].gap}, target_type='collimator', length={ coll_dict[LIN_name]['length']},"+
        f"sigma={sigma_LIN}, jaw_L={line.elements[idx_LIN].jaw_LU })")

    # ---------------------------- TRACKING ----------------------------
    if input_mode == 'pencil_TCP':
        print("\n... Generating initial particles\n")
        # Generate initial pencil distribution on horizontal collimator
        part = xc.generate_pencil_on_collimator(name = TCP_name, line = line, num_particles=num_particles)
        part.at_element = idx_TCP 
        part.start_tracking_at_element = idx_TCP 

    elif input_mode == 'pencil_CRY':
        print("\n... Generating initial particles\n")
        #idx = line.element_names.index(CRY_name)
        part = xc.generate_pencil_on_collimator(name = CRY_name, line=line, num_particles=num_particles)
        idx_monitor = line.element_names.index('CRY_monitor')
        part.at_element = idx_monitor 
        part.start_tracking_at_element = idx_monitor

    elif input_mode == 'circular_halo':
        print("\n... Generating 2D uniform circular sector\n")
        ip1_idx = line.element_names.index('ip1')
        at_s = line.get_s_position(ip1_idx)
        # Vertical plane: generate cut halo distribution
        (x_in_sigmas, px_in_sigmas, r_points, theta_points
            )= xp.generate_2D_uniform_circular_sector(
                                                num_particles=num_particles,
                                                r_range=(gaps[CRY_name] - 0.003,  gaps[CRY_name]+0.002), # sigmas
                                                )
        
        y_in_sigmas, py_in_sigmas = xp.generate_2D_gaussian(num_particles)
        #transverse_spread_sigma = 1 #0.01
        #x_in_sigmas   = np.random.normal(loc=3.45e-7, scale=transverse_spread_sigma, size=num_particles)
        #px_in_sigmas = np.random.normal(scale=transverse_spread_sigma, size=num_particles)

        print(f"Generating {num_particles} particles in a circular sector with sigma = [{gaps[CRY_name]- 0.003}, {gaps[CRY_name]+0.002}] and transverse sigma = 1")

        part = line.build_particles(
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas,
            nemitt_x=normalized_emittance, nemitt_y=normalized_emittance, match_at_s=at_s, at_element=ip1_idx)
            #nemitt_x=normalized_emittance, nemitt_y=normalized_emittance, match_at_s=line.get_s_position(idx_CRY), at_element=idx_CRY)
        
        part.at_element = ip1_idx 
        part.start_tracking_at_element = ip1_idx
 

    # Track
    xc.enable_scattering(line)
    print(line[idx_CRY].critical_angle)
    line.track(part, num_turns=num_turns, time=True)
    xc.disable_scattering(line)
    print(f"Done tracking in {line.time_last_track:.1f}s.")




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

    CRY_imp = impacts.interactions_per_collimator(CRY_name).reset_index()
    n_CRY_abs = CRY_imp['int'].apply(lambda x: 'A'  in x).sum()
    print(f"\nCRY: {n_CRY_abs} particles absorbed\n")
    CRY_imp = CRY_imp.groupby('pid').agg(list).reset_index()[['pid', 'turn']]
    CRY_imp.rename(columns={'turn': 'CRY_turn', 'pid':'particle_id'}, inplace=True)
    df_part = pd.merge(df_part, CRY_imp, on='particle_id', how='left')
    del CRY_imp
    gc.collect()
    
    for column in ['CRY_turn']:
        df_part[column] = df_part[column].apply(lambda x: x if isinstance(x, list) else [])

    print("... Saving metadata\n")
    metadata = {'p0c': line.particle_ref.p0c[0], 'mass0': line.particle_ref.mass0, 'q0': line.particle_ref.q0, 'gamma0': line.particle_ref.gamma0[0], 'beta0': line.particle_ref.beta0[0], 'CRY_absorbed': n_CRY_abs}
    pd.DataFrame(list(metadata.values()), index=metadata.keys()).to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='metadata', format='table', mode='a',
            complevel=9, complib='blosc')


    if 'CRY_impacts' in save_list:
        # SAVE IMPACTS ON CRY
        print("... Saving impacts on CRY\n")

        CRY_monitor_dict = CRY_monitor.to_dict()
        
        ydim_CRY = coll_dict[CRY_name]['width']
        xdim_CRY =  coll_dict[CRY_name]['height']
        jaw_L_CRY = line.elements[idx_CRY].jaw_U
        
        impact_part_df = get_df_to_save(CRY_monitor_dict, df_part, x_dim = xdim_CRY, y_dim = ydim_CRY, jaw_L = jaw_L_CRY, 
                epsilon = epsilon_CRY, num_particles=num_particles, num_turns=num_turns, 
                df_imp = impacts.interactions_per_collimator(CRY_name).reset_index())
        
        del CRY_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='CRY_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()


    if 'LIN_SCAN_impacts' in save_list:

        print("... Saving impacts on TCSG (LIN SCAN)\n")

        LIN_SCAN_monitor_dict = LIN_SCAN_monitor.to_dict()        
        jaw_L_LIN_SCAN= line.elements[idx_LIN].jaw_LU
        
        impact_part_df = get_df_to_save(LIN_SCAN_monitor_dict, df_part, jaw_L = jaw_L_LIN_SCAN, 
                epsilon = epsilon_LIN, num_particles=num_particles, num_turns=num_turns, 
                df_imp = impacts.interactions_per_collimator(LIN_name).reset_index())
        
        del LIN_SCAN_monitor_dict
        gc.collect()

        if output_mode == 'reduced':
            impact_part_df = impact_part_df[['particle_id', 'x', 'px', 'y', 'py', 'this_turn']]

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='LIN_SCAN_impacts', format='table', mode='a',
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