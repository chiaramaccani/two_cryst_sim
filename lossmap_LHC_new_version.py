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



def get_df_to_save(dict, df_part, num_particles, num_turns, epsilon = 0, start = False, x_dim = None, y_dim = None, jaw_L = None):

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

    TCCS_align_angle_step = float(run_dict['TCCS_align_angle_step'])

    normalized_emittance = run_dict['normalized_emittance']

    target_mode = run_dict['target_mode']
    input_mode = run_dict['input_mode']
    load_input_path = run_dict['load_input_path']
    turn_on_cavities = bool(run_dict['turn_on_cavities'])
    print('\nTarget mode: ', target_mode, '\t', 'input mode: ', input_mode, '\t',  'Seed: ', seed, '\tCavities on: ', turn_on_cavities ,  '\n')

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
    TCLA_name = 'tcla.a5l3.b2'

    d_pix = 1 # [m]
    ydim_PIXEL = 0.01408
    xdim_PIXEL = 0.04246

    TCCS_loc = end_s - 6773.7 #6775
    TCCP_loc = end_s - 6653.3 #6655

    dx = 1e-11
    TARGET_loc = end_s - (6653.3 + coll_dict[TCCP_name]["length"]/2 + coll_dict[TARGET_name]["length"]/2 + dx)
    PIXEL_loc = end_s - (6653.3 - coll_dict[TCCP_name]["length"]/2 - d_pix)
    TCP_loc = line.get_s_position()[line.element_names.index(TCP_name)]
    TCLA_loc = line.get_s_position()[line.element_names.index(TCLA_name)]


    line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)
    line.insert_element(at_s=TCCS_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCS_name+'_aper')
    line.insert_element(at_s=TCCP_loc, element=xt.Marker(), name=TCCP_name)
    line.insert_element(at_s=TCCP_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCP_name+'_aper')
    line.insert_element(at_s=TARGET_loc, element=xt.Marker(), name=TARGET_name)
    line.insert_element(at_s=TARGET_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name= TARGET_name + '_aper')
    line.insert_element(at_s=PIXEL_loc, element=xt.Marker(), name=PIXEL_name)
    
    if 'TCCS_impacts' in save_list:
        TCCS_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TCCS_loc - coll_dict[TCCS_name]["length"]/2 - dx, element=TCCS_monitor, name='TCCS_monitor')
        print('\n... TCCS monitor inserted')

    if 'TARGET_impacts' in save_list:
        TARGET_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TARGET_loc - coll_dict[TARGET_name]["length"]/2 - dx, element=TARGET_monitor, name='TARGET_monitor')
        print('\n... TARGET monitor inserted')

    if 'TCCP_impacts' in save_list:
        TCCP_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TCCP_loc - coll_dict[TCCP_name]["length"]/2 - dx/2, element=TCCP_monitor, name='TCCP_monitor')
        print('\n... TCCP monitor inserted')

    if 'PIXEL_impacts' in save_list:
        PIXEL_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = PIXEL_loc, element=PIXEL_monitor, name='PIXEL_monitor')
        print('\n... PIXEL monitor inserted')

    if 'TCP_generated' in save_list:
        TCP_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TCP_loc + coll_dict[TCP_name]["length"]/2 + 1e5*dx, element=TCP_monitor, name='TCP_monitor') 
        print('\n... TCP monitor inserted')

    if 'TCLA_impacts' in save_list:
        TCLA_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = TCLA_loc - coll_dict[TCLA_name]["length"]/2 - 1e5*dx, element=TCLA_monitor, name='TCLA_monitor') 
        print('\n... TCLA monitor inserted')

    if 'BLM_impacts' in save_list:
        BLM_loc = end_s - (6653.3 - 4)
        BLM_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
        line.insert_element(at_s = BLM_loc, element=BLM_monitor, name='BLM_monitor') 
        print('\n... BLM monitor inserted')

    
    
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

        if target_mode == 'target_absorber': 
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
    
    # Printout useful informations
    idx_TCCS = line.element_names.index(TCCS_name)
    idx_TARGET = line.element_names.index(TARGET_name)
    idx_TCCP = line.element_names.index(TCCP_name)
    idx_PIXEL = line.element_names.index(PIXEL_name)
    idx_TCP = line.element_names.index(TCP_name)
    idx_TCLA = line.element_names.index(TCLA_name)

    tw = line.twiss()
    beta_rel = float(line.particle_ref.beta0)
    gamma = float(line.particle_ref.gamma0)
    emittance_phy = normalized_emittance/(beta_rel*gamma)

    sigma_TCCS = np.sqrt(emittance_phy*tw['bety',TCCS_name])
    sigma_TCCP = np.sqrt(emittance_phy*tw['bety',TCCP_name])
    sigma_TARGET = np.sqrt(emittance_phy* tw['bety',TARGET_name])
    sigma_PIXEL = np.sqrt(emittance_phy*tw['bety',PIXEL_name])
    sigma_TCP = np.sqrt(emittance_phy*tw['bety',TCP_name])
    sigma_TCLA = np.sqrt(emittance_phy*tw['bety',TCLA_name])
    
    print(f"\nTCCS\nCrystalAnalysis(n_sigma={round((line.elements[idx_TCCS].jaw_L- tw['y',TCCS_name])/sigma_TCCS, 4)}, length={ coll_dict[ TCCS_name]['length']}, ydim={ coll_dict[ TCCS_name]['xdim']}, xdim={ coll_dict[ TCCS_name]['ydim']}," + 
        f"bending_radius={ coll_dict[ TCCS_name]['bending_radius']}, align_angle={ line.elements[idx_TCCS].align_angle}, sigma={sigma_TCCS}, jaw_L={line.elements[idx_TCCS].jaw_L})")
    print(f"TARGET\nTargetAnalysis(n_sigma={ round((line.elements[idx_TARGET].jaw_L- tw['y',TCCP_name])/sigma_TARGET, 4)}, length={ coll_dict[ TARGET_name]['length']}, ydim={ coll_dict[ TARGET_name]['xdim']}, xdim={ coll_dict[ TARGET_name]['ydim']},"+
        f"sigma={sigma_TARGET}, jaw_L={line.elements[idx_TARGET].jaw_L})")
    print(f"TCCP\nCrystalAnalysis(n_sigma={ round((line.elements[idx_TCCP].jaw_L- tw['y',TARGET_name])/sigma_TCCP, 4)}, length={ coll_dict[ TCCP_name]['length']}, ydim={ coll_dict[ TCCP_name]['xdim']}, xdim={ coll_dict[ TCCP_name]['ydim']},"+ 
        f"bending_radius={ coll_dict[ TCCP_name]['bending_radius']}, align_angle={line.elements[idx_TCCP].align_angle}, sigma={sigma_TCCP}, jaw_L={line.elements[idx_TCCP].jaw_L})")
    print(f"TCP\nTargetAnalysis(n_sigma={ round((line.elements[idx_TCP].jaw_L- tw['y',TCP_name])/sigma_TCP, 4)}, length={coll_dict[ TCP_name]['length']}, ydim={0.025}, xdim={0.025},"+ 
        f"sigma={sigma_TCP}, jaw_L={line.elements[idx_TCP].jaw_L })")
    print(f"TCLA\nTargetAnalysis(n_sigma={round(( line.elements[idx_TCLA].jaw_L - tw['y',TCLA_name])/sigma_TCLA, 4)}, length={coll_dict[ TCLA_name]['length']}, ydim={0.025}, xdim={0.025},"+ 
        f"sigma={sigma_TCLA},  jaw_L={line.elements[idx_TCLA].jaw_L})")
    print(f"PIXEL\nTargetAnalysis(n_sigma={PIXEL_gap}, length={0}, ydim={ydim_PIXEL}, xdim={xdim_PIXEL},"+ 
        f"sigma={sigma_PIXEL})\n")
        
    if "BLM_impacts" in save_list:
        idx_BLM = line.element_names.index('BLM_monitor')
        sigma_BLM = np.sqrt(emittance_phy*tw['bety','BLM_monitor'])
        print(f"BLM\nTargetAnalysis(n_sigma={0.03/sigma_BLM}, length={0}, ydim={0.025}, xdim={0.025}, sigma={sigma_BLM}, jaw_L={0.03})\n")



    # ---------------------------- TRACKING ----------------------------
    # Generate initial pencil distribution on horizontal collimator
    tcp  = f"tcp.{'c' if plane=='H' else 'd'}6{'l' if beam=='1' else 'r'}7.b{beam}"
    idx = line.element_names.index(tcp)
    
    if input_mode == 'generate':
        print("\n... Generating initial particles\n")
        part = xc.generate_pencil_on_collimator(line, tcp, num_particles, nemitt_x=normalized_emittance, nemitt_y=normalized_emittance)
        part.at_element = idx 
        part.start_tracking_at_element = idx 
        #process=psutil.Process(os.getpid())
        #print(process.memory_info().rss)
        
        
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
        #process=psutil.Process(os.getpid())
        #print(process.memory_info().rss)
        del df_part, dct_part
        gc.collect()
        #print(process.memory_info().rss)
        part.at_element = idx + 2 
        part.start_tracking_at_element = idx + 2

    save_inital_particles = False
    if save_inital_particles:
        print("\n... Saving initial particles\n")
        part.to_pandas().to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='initial_particles', format='table', mode='a',
            complevel=9, complib='blosc')

    

    # Track
    coll_manager.enable_scattering()
    line.track(part, num_turns=num_turns, time=True)
    coll_manager.disable_scattering()
    print(f"Done tracking in {line.time_last_track:.1f}s.")

    # Printout useful informations
    print("\n----- Check information -----")
    print(f"Line index of TCCS: {idx_TCCS}, TARGET: {idx_TARGET}, TCCP: {idx_TCCP}, PIXEL: {idx_PIXEL}, TCP: {idx_TCP}\n")
    


    # ---------------------------- LOSSMAPS ----------------------------    
    line_is_reversed = True if f'{beam}' == '2' else False
    ThisLM = xc.LossMap(line, line_is_reversed=line_is_reversed, part=part)
    print(ThisLM.summary)

    if 'losses' in save_list:
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
    #df_part.to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='particles', format='table', mode='a',
    #          complevel=9, complib='blosc')



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
        
        ydim_TCCS = coll_dict[TCCS_name]['xdim']
        xdim_TCCS =  coll_dict[TCCS_name]['ydim']
        jaw_L_TCCS = line.elements[idx_TCCS].jaw_L
        
        impact_part_df = get_df_to_save(TCCS_monitor_dict, df_part, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = jaw_L_TCCS, 
                epsilon = 0, num_particles=num_particles, num_turns=num_turns)
        
        del TCCS_monitor_dict
        gc.collect()

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCCS_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()


    if 'TCCP_impacts' in save_list:
        # SAVE IMPACTS ON TCCP
        print("... Saving impacts on TCCP\n")

        TCCP_monitor_dict = TCCP_monitor.to_dict()
        
        ydim_TCCP = coll_dict[TCCP_name]['xdim']
        xdim_TCCP =  coll_dict[TCCP_name]['ydim']
        jaw_L_TCCP = line.elements[idx_TCCP].jaw_L
        
        impact_part_df = get_df_to_save(TCCP_monitor_dict, df_part, x_dim = xdim_TCCP, y_dim = ydim_TCCP, jaw_L = jaw_L_TCCP, 
                epsilon = 0.5e-3, num_particles=num_particles, num_turns=num_turns)

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCCP_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        embed()
        
        del impact_part_df
        gc.collect()
        

    if 'TARGET_impacts' in save_list:

        # SAVE IMPACTS ON TARGET
        print("... Saving impacts on TARGET\n")

        TARGET_monitor_dict = TARGET_monitor.to_dict()
       
        ydim_TARGET = coll_dict[TARGET_name]['xdim']
        xdim_TARGET =  coll_dict[TARGET_name]['ydim']
        jaw_L_TARGET = line.elements[idx_TARGET].jaw_L 

        impact_part_df = get_df_to_save(TARGET_monitor_dict, df_part, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = jaw_L_TARGET,
                epsilon = 2.5e-3, num_particles=num_particles, num_turns=num_turns)
        
        del TARGET_monitor_dict
        gc.collect()

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TARGET_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        

    if 'PIXEL_impacts' in save_list:

        # SAVE IMPACTS ON PIXEL
        print("... Saving impacts on PIXEL\n")

        PIXEL_monitor_dict = PIXEL_monitor.to_dict()
    
        jaw_L_PIXEL = 0.008 #sigma_PIXEL * PIXEL_gap        

        impact_part_df = get_df_to_save(PIXEL_monitor_dict, df_part,  jaw_L = jaw_L_PIXEL,  #x_dim = xdim_PIXEL, y_dim = ydim_PIXEL,
                epsilon = 2.5e-3, num_particles=num_particles, num_turns=num_turns)
        
        del PIXEL_monitor_dict
        gc.collect()

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='PIXEL_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()


    
    
    if 'TCLA_impacts' in save_list:

        # SAVE IMPACTS ON PIXEL
        print("... Saving impacts on TCLA\n")

        TCLA_monitor_dict = TCLA_monitor.to_dict()
    
        jaw_L_TCLA = line.elements[idx_TCLA].jaw_L 

        impact_part_df = get_df_to_save(TCLA_monitor_dict, df_part,  jaw_L = jaw_L_TCLA,
                num_particles=num_particles, num_turns=num_turns, epsilon = 2.5e-3)
        
        del TCLA_monitor
        gc.collect()

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='TCLA_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()


    if 'BLM_impacts' in save_list:

        # SAVE IMPACTS ON PIXEL
        print("... Saving impacts on BLM\n")

        BLM_monitor_dict = BLM_monitor.to_dict()
    
        jaw_L_BLM = 0.03          

        impact_part_df = get_df_to_save(BLM_monitor_dict, df_part,  jaw_L = jaw_L_BLM,
                num_particles=num_particles, num_turns=num_turns, epsilon = 0)
        
        del BLM_monitor
        gc.collect()

        impact_part_df.to_hdf(Path(path_out, f'particles_B{beam}{plane}.h5'), key='BLM_impacts', format='table', mode='a',
            complevel=9, complib='blosc')
        
        del impact_part_df
        gc.collect()



if __name__ == "__main__":
    main()
