import json
import os
import subprocess
import sys

import numpy as np
from pathlib import Path
import yaml
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator
import lossmaps as lm
import xobjects as xo

import xtrack as xt
import xcoll as xc

import lossmaps as lm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import xarray as xr







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
    print('vmabc.4l2.b.b2_aper' in aperture_offsets.keys())
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





def main():

    config_file = sys.argv[1]
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)


    name = sys.argv[1]

    run_dict = config_dict['run']
    file_dict = config_dict['input_files']


    context = xo.ContextCpu(omp_num_threads='auto')

    # On a modern CPU, we get ~5000 particle*turns/s
    # So this script should take around half an hour
    beam          = run_dict['beam']
    plane         = run_dict['plane']

    num_turns     = run_dict['turns']
    num_particles = run_dict['nparticles']
    engine        = run_dict['engine']

    TTCS_align_angle_step = run_dict['TTCS_align_angle_step']

    mode = 'normal' #run_dict['mode']


    path_out = Path.cwd() / 'Outputdata'

    if not path_out.exists():
        os.makedirs(path_out)


    # Load from json
    line = xt.Line.from_json(file_dict[f'line_b{beam}'])

    end_s = line.get_length()


    TCCS_loc = end_s - 6773.7 #6775
    TCCP_loc = end_s - 6653.3 #6655
    TARGET_loc = end_s - (6653.3 + 0.07/2 +0.005/2)


    line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name='tccs.5r3.b2')
    line.insert_element(at_s=TCCS_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name='tccs.5r3.b2_aper')
    line.insert_element(at_s=TCCP_loc, element=xt.Marker(), name='tccp.4l3.b2')
    line.insert_element(at_s=TCCP_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name='tccp.4l3.b2_aper')
    line.insert_element(at_s=TARGET_loc, element=xt.Marker(), name='target.4l3.b2')
    line.insert_element(at_s=TARGET_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name='target.4l3.b2_aper')

    #line.cycle(name_first_element='ip3', inplace=True)

    bad_aper = find_bad_offset_apertures(line)
    print('Bad apertures : ', bad_aper)
    print('Replace bad apertures with Marker')
    for name in bad_aper.keys():
        line.element_dict[name] = xt.Marker()
        print(name, line.get_s_position(name), line.element_dict[name])



    # Aperture model check
    print('\nAperture model check on imported model:')
    df_imported = line.check_aperture()
    assert not np.any(df_imported.has_aperture_problem)




    # Initialise collmanager
    coll_manager = xc.CollimatorManager.from_yaml(file_dict['collimators'], line=line, beam=beam, _context=context, ignore_crystals=False)

    #print(coll_manager.collimator_names)

    # Install collimators into line
    if engine == 'everest':
        coll_names = coll_manager.collimator_names

        if mode == 'cry_black_absorbers':
            black_absorbers = ['target.4l3.b2', 'tccs.5r3.b2']
        elif mode == 'angular_scan': 
            black_absorbers = ['target.4l3.b2',]
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
    coll_manager.build_tracker()


    # Set the collimator openings based on the colldb,
    # or manually override with the option gaps={collname: gap}
    coll_manager.set_openings()


    TTCS_name = 'tccs.5r3.b2'

    if mode == 'angular_scan':
        print("\nTTCS aligned to beam: ", line[TTCS_name].align_angle)
        #line[TTCS_name].align_angle = TTCS_align_angle_step

        line[TTCS_name].align_angle = line[TTCS_name].align_angle + TTCS_align_angle_step
        print("TTCS align angle incremented by step: ", line[TTCS_name].align_angle)


    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    #assert not np.any(df_with_coll.has_aperture_problem)


    num_turns     = 150 #run_dict['turns']
    num_particles = 10 #run_dict['nparticles']

        # Generate initial pencil distribution on horizontal collimator
    tcp  = f"tcp.{'c' if plane=='H' else 'd'}6{'l' if beam=='1' else 'r'}7.b{beam}"
    part = coll_manager.generate_pencil_on_collimator(tcp, num_particles=num_particles)


    # Optimise the line
    #line.optimize_for_tracking()
    idx = line.element_names.index(tcp)
    part.at_element = idx
    part.start_tracking_at_element = idx


    # Track
    coll_manager.enable_scattering()

    #df = pd.DataFrame(columns=['turn', 'x', 'y', 'px', 'py', 'state', 'at_element'])

    #part_array = xr.DataArray(np.zeros((6,num_particles, num_turns)), coords=[['x', 'y', 'px', 'py', 'state','at_element'], np.arange(num_particles), np.arange(num_turns)], dims=['properties', 'id', 'turns'])
    part_array = xr.DataArray(np.zeros((num_particles, 6, num_turns)), coords=[np.arange(num_particles),['x', 'y', 'px', 'py', 'state','at_element'], np.arange(num_turns)], dims=['id', 'properties', 'turns'])

    for n in range(num_turns):
        line.track(part, num_turns=1, time=True)
        #new_row = pd.DataFrame({ 'x':part.x, 'y':part.y, 'px':part.px, 'py':part.py, 'state':part.state,'at_element':part.at_element})
        #turn_info = xr.DataArray(np.concatenate((part.x, part.y, part.px, part.py, part.state, part.at_element)).reshape(6, num_particles), coords=[['x', 'y', 'px', 'py', 'state','at_element'],np.arange(num_particles)], dims=['properties', 'id'])
        turn_info = xr.DataArray(np.column_stack((part.x, part.y, part.px, part.py, part.state, part.at_element)).reshape(num_particles, 6), coords=[np.arange(num_particles), ['x', 'y', 'px', 'py', 'state','at_element']], dims=['id', 'properties'])
        
        part_array[:,:,n] = turn_info
        #df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)
    
    
    coll_manager.disable_scattering()
    print(f"Done tracking in {line.time_last_track:.1f}s.")


    # Save lossmap to json, which can be loaded, combined (for more statistics),
    # and plotted with the 'lossmaps' package
    _ = coll_manager.lossmap(part, file=Path(path_out,f'lossmap_B{beam}{plane}.json'))


    # Save a summary of the collimator losses to a text file
    summary = coll_manager.summary(part)
    print(summary)

    name = sys.argv[1]
    part_array.to_netcdf("./Outputdata/"+name+".nc")



def plot_distributions_at_turn(arr_path, turn):
    
    arr =  xr.open_dataarray(arr_path)
    alive_arr = arr.where(arr.loc[:, 'state', turn] > 0, drop = True)[:,:,turn]

    fig1 = plt.figure( figsize=(24, 5))
    ax1 = fig1.add_subplot(1,3,1)
    ax1.hist(alive_arr.loc[:, 'x'], bins=100)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel("")
    #ax1.set_yscale("log")
    ax1.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e3:.{1}f}" for x in plt.xticks()[0]])


    ax2 = fig1.add_subplot(1,3,2)
    ax2.hist(alive_arr.loc[:, 'y'], bins=100) 
    ax2.set_xlabel('y [mm]')
    ax2.set_ylabel('')
    #ax2.set_yscale("log")
    ax2.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e3:.{1}f}" for x in plt.xticks()[0]])
    ax2.set_title(f'Total particles: {alive_arr.shape[0]}')

    ax3 = fig1.add_subplot(1,3,3)
    h = ax3.hist2d(alive_arr.loc[:, 'x'], alive_arr.loc[:, 'y'], bins=100, norm=matplotlib.colors.LogNorm())#vmin = 1, vmax = 1e6, range = ([-40e-6, 40e-6], [-40e-6,40e-6])) 
    ax3.set_xlabel(r'x [mm]')
    #ax3.set_ylim(0,0.008)
    ax3.set_ylabel(r'y [mm]')
    ax3.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e3:.{1}f}" for x in plt.xticks()[0]])
    ax3.set_yticks(ticks=plt.yticks()[0], labels=[f"{y*1e3:.{1}f}" for y in plt.yticks()[0]])

    axins = inset_axes(ax3, height="100%",  width="5%", loc='right', borderpad=-6 )
    fig1.colorbar(h[3], cax=axins, orientation='vertical', label='Count (log scale)')
    ax3.grid(linestyle=':')

    

    


    """ax3_tw = ax3.twinx()
    ticks = np.arange(1, max(ax3.get_yticks())/sigma+1, 2.0)
    ax3_tw.set_yticks(ticks)
    ax3_tw.set_ylabel(r' n $\sigma$')
    ax3_tw.set_yticklabels([f"{x /sigma :.{0}f}"  for x in ticks])
    ax3_tw.axhline(5, color = 'r', linestyle = '--')
    ax3_tw.text( max(ax3.get_xticks())-1.5e-3, 4, r'TCP $\sigma$')"""
    

    fig1.suptitle('plot')
    plt.show()


    return fig1, [ax1,ax2,ax3]




if __name__ == "__main__":
    main()