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

    if plane == 'DPpos' or plane == 'DPneg':
        sweep = 300
        sweep = -abs(sweep) if plane == 'DPpos' else abs(sweep)
        num_turns     = int(20*abs(sweep))
    num_particles = run_dict['nparticles']
    print('\nNumber of particles: ', num_particles, ' number of turns: ', num_turns, '\n')

    fixed_seed    = bool(run_dict['fixed_seed'])
    seed          = run_dict['seed']

    adt_amplitude    =  None if run_dict['adt_amplitude']== 'None' else float(run_dict['adt_amplitude'])

    if fixed_seed:
        np.random.seed(seed=seed)
        print('\n----- Seed set to: ', seed)

    TCCS_align_angle_step = float(run_dict['TCCS_align_angle_step'])
    TCCP_align_angle_step = float(run_dict['TCCP_align_angle_step'])
    TCCP_align_angle_additional = float(run_dict['TCCP_align_angle_additional']) if 'TCCP_align_angle_additional' in run_dict.keys() else 0

    normalized_emittance = run_dict['normalized_emittance']

    target_mode = run_dict['target_mode']
    input_mode = run_dict['input_mode']
    output_mode = run_dict['output_mode']
    turn_on_cavities = bool(run_dict['turn_on_cavities'])
    print('\nTarget mode: ', target_mode, '\t', 'input mode: ', input_mode, '\t', 'output mode: ', output_mode, '\t',  'Seed: ', seed, '\tCavities on: ', turn_on_cavities , '\t ADT: ', adt_amplitude, '\n')

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



    if beam == "2":
        TCCS_name = 'tccs.5r3.b2'
        TCCP_name = 'tccp.4l3.b2'
        TARGET_name = 'target.4l3.b2'
        PIXEL_name = 'pixel.detector'
        TFT_name = 'tft.detector'
        TCP_name = 'tcp.d6r7.b2'
        TCLA_name = 'tcla.a5l3.b2'

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

        TCSG_1_loc = line.get_s_position()[line.element_names.index('tcsg.4l3.b2')] + 1 
        TCSG_2_loc = line.get_s_position()[line.element_names.index('tcsg.a5l3.b2')] + 1
        TCSG_3_loc = line.get_s_position()[line.element_names.index('tcsg.b5l3.b2')] + 1

        line.insert_element(at_s=TCSG_1_loc, element=xt.Marker(), name='tcsg.4l3_ala.b2')
        line.insert_element(at_s=TCSG_1_loc, element=xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a=0.04, b=0.04) , name='tcsg.4l3_ala.b2' + '_aper')
        line.insert_element(at_s=TCSG_2_loc, element=xt.Marker(), name='tcsg.a5l3_ala.b2')
        line.insert_element(at_s=TCSG_2_loc, element=xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a=0.04, b=0.04) , name='tcsg.a5l3_ala.b2' + '_aper')
        line.insert_element(at_s=TCSG_3_loc, element=xt.Marker(), name='tcsg.b5l3_ala.b2')
        line.insert_element(at_s=TCSG_3_loc, element=xt.LimitRectEllipse(max_x=0.04, max_y=0.04, a=0.04, b=0.04) , name='tcsg.b5l3_ala.b2' + '_aper')
        
        energy_gev = line.particle_ref.p0c/1e9
        line['mcbwv.4l3.b2'].knl = 0.2998 * 1.1 / energy_gev



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

    embed()


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
    if beam == "2":
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
            for pix_idx in ['_1', '_2', '_3']:
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



    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)



    # ---------------------------- CALCULATE TWISS ----------------------------
    tw = line.twiss()

    if adt_amplitude is not None:
        if plane == 'H':
            adt.calibrate_by_emittance(nemitt=normalized_emittance, twiss=tw)
        else:
            adt.calibrate_by_emittance(nemitt=normalized_emittance, twiss=tw)


    line.discard_tracker()

    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    TCP_name =  f"tcp.{'c' if plane=='H' else 'd'}6{'l' if f'{beam}'=='1' else 'r'}7.b{beam}"
    idx_TCP = line.element_names.index(TCP_name)

    # ---------------------------- INPUT GENERATION ----------------------------
    if input_mode == 'pencil_TCP':
        # Generate initial pencil distribution on horizontal collimator
        TCP_name =  f"tcp.{'c' if plane=='H' else 'd'}6{'l' if f'{beam}'=='1' else 'r'}7.b{beam}"
        print("\n... Generating initial particles on TCP: {tcp} \n")
        idx_TCP = line.element_names.index(TCP_name)
        impact_parameter = 0 
        part = line[TCP_name].generate_pencil(num_particles = num_particles, impact_parameter = impact_parameter)
        #part = xc.generate_pencil_on_collimator(name = TCP_name, line = line, num_particles=num_particles)
        part.at_element = idx_TCP 
        part.start_tracking_at_element = idx_TCP 


    elif input_mode == 'circular_halo':
        print("\n... Generating 2D uniform circular sector\n")
        ip1_idx = line.element_names.index('ip1')
        at_s = line.get_s_position(ip1_idx)
        gap = line[TCP_name].gap
        print('TCP gap: ', gap)
        # Vertical plane: generate cut halo distribution
        (y_in_sigmas, py_in_sigmas, r_points, theta_points
            )= xp.generate_2D_uniform_circular_sector(
                                                num_particles=num_particles,
                                                r_range=(gap - 0.003, gap+0.002), # sigmas
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

    elif input_mode == 'matched_gaussian':
        print("\n... Generating matched gaussian\n")
        part = xp.generate_matched_gaussian_bunch(nemitt_x=colldb.nemitt_x,
                                          nemitt_y=colldb.nemitt_y,
                                          sigma_z=7.55e-2, num_particles=num_particles, line=line)


    # ---------------------------- SEED FIXING ----------------------------
  
    if fixed_seed:
        print("\n... Fixing seed of particles\n")
        random_array = np.random.randint(0, 4291630464,  size = num_particles*4)
        part._rng_s1 = random_array[0:num_particles]
        part._rng_s2 = random_array[num_particles:num_particles*2]
        part._rng_s3 = random_array[num_particles*2:num_particles*3]
        part._rng_s4 = random_array[num_particles*3:num_particles*4]

    if plane == 'DPpos' or plane == 'DPneg':
        rf_sweep = xc.RFSweep(line)
        rf_sweep.info(sweep=sweep, num_turns=num_turns)


    # ---------------------------- TRACKING ----------------------------
    #line.optimize_for_tracking()
    line.scattering.enable() 

    if adt_amplitude is not None:
        adt.activate()

    if plane == 'H' or plane == 'V':
        line.track(part, num_turns=num_turns, time=True)
    elif plane == 'DPpos' or plane == 'DPneg':
        rf_sweep.track(sweep=sweep, particles=part, num_turns=num_turns, time=True, with_progress=5)

    if adt_amplitude is not None:
        adt.deactivate()

    line.scattering.disable()
    print(f"\nDone tracking in {line.time_last_track:.1f}s.")


    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu())
    line_is_reversed = True if f'{beam}' == '2' else False
    ThisLM = xc.LossMap(line, line_is_reversed=line_is_reversed, part=part)
    print(ThisLM.summary, '\n')
    ThisLM.to_json(file=Path(path_out, f'lossmap_B{beam}{plane}.json'))
    #ThisLM.save_summary(file=Path(path_out, f'coll_summary_B{beam}{plane}.out'))

    


if __name__ == "__main__":
    main()





