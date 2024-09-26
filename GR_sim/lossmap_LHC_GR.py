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
import gc
import io 
#import psutil

from IPython import embed





# ---------------------------- LOADING FUNCTIONS ----------------------------

def get_df_to_save(dict, df_part, num_particles, num_turns, epsilon = 0, start = False, jaw_L = None, plane = 'V'):

    float_variables = ['zeta', 'x', 'px', 'y', 'py', 'delta', 'p0c']
    int_variables = ['at_turn', 'particle_id', 'at_element', 'state', 'parent_particle_id']
    variables = float_variables + int_variables
    variables.remove('at_element')

    var_dict = {}

    for var in variables:
        var_dict[var] = dict['data'][var].reshape((num_particles, num_turns))   
    del dict
    gc.enable()
    gc.collect()

    
    impact_part_dict = {}
    for key in var_dict.keys():
        impact_part_dict[key] = []

    if jaw_L is not None:
        if plane == 'V':
            abs_y_low = jaw_L
            for part in range(num_particles):
                for turn in range(num_turns):
                    if var_dict['state'][part, turn] > 0 and var_dict['y'][part, turn] > (abs_y_low - epsilon):
                        for key in var_dict.keys():
                            impact_part_dict[key].append(var_dict[key][part, turn])
        elif plane == 'H':
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
    
    impact_part_df.rename(columns={'state': 'this_state'}, inplace=True)
    impact_part_df.rename(columns={'at_turn': 'this_turn'}, inplace=True)
    impact_part_df = pd.merge(impact_part_df, df_part[['at_element', 'state', 'at_turn', 'particle_id']], on='particle_id', how='left')
    
    impact_part_df[float_variables] = impact_part_df[float_variables].astype('float32')
    impact_part_df[int_variables] = impact_part_df[int_variables].astype('int32')
    impact_part_df['this_turn'] = impact_part_df['this_turn'].astype('int32')
    impact_part_df['this_state'] = impact_part_df['this_state'].astype('int32')

    
    return impact_part_df
    



# ---------------------------- MAIN ----------------------------


def main():

    config_file = sys.argv[1]
    
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # Configure run parameters
    run_dict = config_dict['run']

    beam          = run_dict['beam']
    plane         = run_dict['plane']

    num_turns     = run_dict['turns']
    num_particles = run_dict['nparticles']
    engine        = run_dict['engine']
    
    seed          = run_dict['seed']

    normalized_emittance = run_dict['normalized_emittance']

    input_mode = run_dict['input_mode']

    lim_max_B2H = float(run_dict['lim_max_B2H'])
    lim_min_B2H = float(run_dict['lim_min_B2H'])
    lim_max_B2V = float(run_dict['lim_max_B2V']) 
    lim_min_B2V = float(run_dict['lim_min_B2V']) 

    turn_on_cavities = bool(run_dict['turn_on_cavities'])
    print('input mode: ', input_mode, '\t',  'Seed: ', seed, '\tCavities on: ', turn_on_cavities ,  '\n')

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
    

    context = xo.ContextCpu(omp_num_threads='auto')

    # Define output path
    path_out = Path.cwd() / 'Outputdata'

    if not path_out.exists():
        os.makedirs(path_out)



    # ---------------------------- SETUP LINE ----------------------------

    # Load from json
    line = xt.Line.from_json(line_file)
    end_s = line.get_length()

    # switch on cavities
    if turn_on_cavities:
        speed = line.particle_ref._xobject.beta0[0]*scipy.constants.c
        harmonic_number = 35640
        voltage = 12e6/len(line.get_elements_of_type(xt.Cavity)[1])
        frequency = harmonic_number * speed /line.get_length()
        for side in ['l', 'r']:
            for cell in ['a','b','c','d']:
                line[f'acsca.{cell}5{side}4.b{beam}'].voltage = voltage
                line[f'acsca.{cell}5{side}4.b{beam}'].frequency = frequency   

    # Aperture model check
    print('\nAperture model check on imported model:')
    df_imported = line.check_aperture()
    assert not np.any(df_imported.has_aperture_problem)


    # Initialise collmanager
    colldb = xc.CollimatorDatabase.from_yaml(coll_file, beam=beam, ignore_crystals=False)
    coll_names = colldb.collimator_names
    #embed()

    # Install collimators into line
    if engine == 'everest':
        
        black_absorbers = []
        everest_colls = [name for name in coll_names if name not in black_absorbers]

        colldb.install_everest_collimators(line = line, names=everest_colls,verbose=True)
        colldb.install_black_absorbers(line = line, names = black_absorbers, verbose=True)
    else:
        raise ValueError(f"Unknown scattering engine {engine}!")



    if 'beta_impacts' in save_list:
        if beam == '1' and plane == 'H':
            beam_list = ['B1H']
        elif beam == '1' and plane == 'V':
            beam_list = ['B1V']
        elif beam == '2' and plane == 'H':
            beam_list = ['B2H']
            lim_max = lim_max_B2H
            lim_min = lim_min_B2H
        elif beam == '2' and plane == 'V':
            beam_list = ['B2V']
            lim_max = lim_max_B2V
            lim_min = lim_min_B2
            
        MAX_BETA_name = 'max.ir3.beta'
        MIN_BETA_name = 'min.ir3.beta'

        if 'B2V' in beam_list:
            MAX_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s - 6626.128995520019), element=MAX_BETA_monitor, name=MAX_BETA_name)
            MIN_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s - 6528.569595520017), element=MIN_BETA_monitor, name=MIN_BETA_name)
            print('\n... B2V monitors inserted')

        if 'B2H' in beam_list:
            MAX_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s - 6529.3983955200165), element=MAX_BETA_monitor, name=MAX_BETA_name)
            MIN_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s - 6624.471395520017 ), element=MIN_BETA_monitor, name=MIN_BETA_name)
            print('\n... B2H monitor inserted')

        if 'B1H' in beam_list:
            MAX_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s - 6800.043196084 ), element=MAX_BETA_monitor, name=MAX_BETA_name)
            MIN_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s - 6704.970196083999 ), element=MIN_BETA_monitor, name=MIN_BETA_name)
            print('\n... B1H monitor inserted')

        if 'B1V' in beam_list:
            MAX_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s -  6703.312596084), element=MAX_BETA_monitor, name=MAX_BETA_name)
            MIN_BETA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
            line.insert_element(at_s = (end_s - 6800.871996084), element=MIN_BETA_monitor, name=MIN_BETA_name)
            print('\n... B1V monitor inserted')

    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)

    # Build the tracker
    line.build_tracker()

    # Set the collimator openings based on the colldb,
    # or manually override with the option gaps={collname: gap}
    xc.assign_optics_to_collimators(line=line)

    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)
    
    if 'beta_impacts' in save_list:
        # Printout useful informations
        idx_MAX_BETA = line.element_names.index(MAX_BETA_name)
        idx_MIN_BETA = line.element_names.index(MIN_BETA_name)

        tw = line.twiss()
        beta_rel = float(line.particle_ref.beta0)
        gamma = float(line.particle_ref.gamma0)
        emittance_phy = normalized_emittance/(beta_rel*gamma)

        if plane == 'V':
            sigma_MAX_BETA = np.sqrt(emittance_phy*tw['bety',idx_MAX_BETA])
            sigma_MIN_BETA = np.sqrt(emittance_phy*tw['bety',idx_MIN_BETA])
            co_ref_max = tw['y',idx_MAX_BETA]
            co_ref_min = tw['y',idx_MIN_BETA]
        elif plane == 'H':
            sigma_MAX_BETA = np.sqrt(emittance_phy*tw['betx',idx_MAX_BETA])
            sigma_MIN_BETA = np.sqrt(emittance_phy*tw['betx',idx_MIN_BETA])
            co_ref_max = tw['x',idx_MAX_BETA]
            co_ref_min = tw['x',idx_MIN_BETA]
        
        print(f"MAX_BETA\nTargetAnalysis(n_sigma={0}, length={0}, xdim={0.25},  ydim={0.25},"+
            f"sigma={sigma_MAX_BETA}, jaw_L={co_ref_max})")
        print(f"MIN_BETA\nTargetAnalysis(n_sigma={0}, length={0}, ydim={0.25}, xdim={0.25},"+ 
            f"sigma={sigma_MIN_BETA}, jaw_L={co_ref_min})")

    # ---------------------------- TRACKING ----------------------------
    # Generate initial pencil distribution on horizontal collimator
    tcp  = f"tcp.{'c' if plane=='H' else 'd'}6{'l' if beam=='1' else 'r'}7.b{beam}"
    idx = line.element_names.index(tcp)
    
    if input_mode == 'generate':
        print("\n... Generating initial particles\n")
        part = xc.generate_pencil_on_collimator(name = tcp, line = line, num_particles=num_particles)
        part.at_element = idx 
        part.start_tracking_at_element = idx     
    

    # Track
    xc.enable_scattering(line)
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
    #print("... Saving impacts on particle data\n")
    df_part = part.to_pandas()
    drop_list = ['chi', 'charge_ratio', 'pdg_id', 'rvv', 'rpp', '_rng_s1', '_rng_s2', '_rng_s3', '_rng_s4', 'weight', 'ptau', 'q0','gamma0','beta0', 'mass0', 'start_tracking_at_element', 's']
    float_variables = ['zeta', 'x', 'px', 'y', 'py', 'delta', 'p0c']
    int_variables = ['at_turn', 'particle_id', 'at_element', 'state', 'parent_particle_id']
    df_part.drop(drop_list, axis=1, inplace=True)
    df_part[float_variables] = df_part[float_variables].astype('float32')
    df_part[int_variables] = df_part[int_variables].astype('int32')


    if 'beta_impacts' in save_list:

        # Printout useful informations
        print("\n----- Check information -----")
        print(f"Line index of MAX: {idx_MAX_BETA}, MIN: {idx_MIN_BETA}\n")


        print("\n... Saving particles at MAX beta location in ", beam_list[0], "\n")

        MAX_BETA_monitor_dict = MAX_BETA_monitor.to_dict()

        impact_part_df = get_df_to_save(MAX_BETA_monitor_dict, df_part, jaw_L = lim_max, plane = plane,
                                        num_particles=num_particles, num_turns=num_turns)

        del MAX_BETA_monitor_dict
        gc.collect()
        
        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='MAX_BETA', format='table', mode='a',
            complevel=9, complib='blosc')

        del impact_part_df
        gc.collect()

        
        print("\n... Saving particles at MIN beta location ", beam_list[0], "\n")

        MIN_BETA_monitor_dict = MIN_BETA_monitor.to_dict()

        impact_part_df = get_df_to_save(MIN_BETA_monitor_dict, df_part, jaw_L = lim_min, plane = plane,
                                        num_particles=num_particles, num_turns=num_turns)

        del MIN_BETA_monitor_dict
        gc.collect()
        
        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='MIN_BETA', format='table', mode='a',
            complevel=9, complib='blosc')

        del impact_part_df
        gc.collect()




if __name__ == "__main__":
    main()

