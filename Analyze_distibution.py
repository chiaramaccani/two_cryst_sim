
import sys
from pathlib import Path
import yaml
import numpy as np
import os
import lossmaps as lm
import subprocess 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json
    
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import xobjects as xo
import xtrack as xt
import xpart as xp
import xcoll as xc





#   ----------------------------------  SETUP LINE    --------------------------------------------------------------------------



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



def setup_line(config_file = 'config_sim.yaml'):
    
    
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)



    sub_dict = config_dict['run']
    file_dict = config_dict['input_files']


    context = xo.ContextCpu(omp_num_threads='auto')

    # On a modern CPU, we get ~5000 particle*turns/s
    # So this script should take around half an hour
    beam          = sub_dict['beam']
    plane         = sub_dict['plane']

    num_turns     = sub_dict['turns']
    num_particles = sub_dict['nparticles']
    engine        = sub_dict['engine']

    TTCS_align_angle_step = sub_dict['TTCS_align_angle_step']



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
        print(line.element_dict[name])



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
        black_absorbers = ['target.4l3.b2', 'tccs.5r3.b2']
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

    coll_manager.set_openings()

    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    #assert not np.any(df_with_coll.has_aperture_problem)

    return coll_manager.line







#   ----------------------------------   CRYSTAL ANGULAR DISTRIBUTION     --------------------------------------------------------------------------

def angular_distribution(prefix_name = 'TEST_B2V_ABS_CRY1_5.5', config_file = 'config_sim.yaml'):

    Condor_path = '/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim/Condor/'
    test_list = [Condor_path + i for i in os.listdir(Condor_path) if prefix_name in i]
    
    for test_name in test_list:

        job_directories = [test_name + '/' + i for i in os.listdir(test_name) if 'Job.' in i]
        n_jobs = len(job_directories)  
        df_particles =  pd.DataFrame()

        n_jobs_verify = 0 

        for job_dir in job_directories:
            if os.path.exists(job_dir +'/Outputdata'):
                part_file = [filename for filename in os.listdir(job_dir +'/Outputdata') if filename.startswith("particles_")]
                if part_file:
                    df_tmp = pd.read_hdf(job_dir +'/Outputdata/' + part_file[0])
                    df_particles = pd.concat([df_particles, df_tmp])
                    n_jobs_verify = n_jobs_verify + 1 


        if n_jobs != n_jobs_verify:
            print("!!! Succesful Jobs: ", n_jobs_verify, '/', n_jobs, ' in file: ', test_name)
        print("Loading particles: done! ", len(df_particles))

        TCCS_name = 'tccs.5r3.b2'

        line_coll = setup_line(config_file)
        TCCS_idx = line_coll.element_names.index(TCCS_name)
        cry_impact_parts = df_particles[(df_particles.at_element == TCCS_idx) & (df_particles.state<0)]

        twiss = line_coll.twiss()

        beta_y_optics = twiss['bety',TCCS_name]
        alfa_y_optics = twiss['alfy',TCCS_name]

        """with open(config_file, 'r') as stream:
            config_dict = yaml.safe_load(stream)

        file_dict = config_dict['input_files']
        sub_dict = config_dict['run']
        beam          = sub_dict['beam']
        plane         = sub_dict['plane']"""

        critical_angle = np.sqrt(2*16/(line_coll.particle_ref._xobject.p0c[0]*line_coll.particle_ref._xobject.beta0[0]))

        normalized_emittance = 3.5e-6
        emittance_phy = normalized_emittance/(line_coll.particle_ref._xobject.beta0[0]*line_coll.particle_ref._xobject.gamma0[0])
        sigma = np.sqrt(emittance_phy*beta_y_optics)

        n_sig = prefix_name.split('CRY1_')[1]

        print('n sigma: ', n_sig)

        py_central = - float(n_sig) * alfa_y_optics * np.sqrt(emittance_phy/beta_y_optics)

        fig1 = plt.figure( figsize=(24, 10))

        ax1 = fig1.add_subplot(2,3,1)
        ax1.hist(cry_impact_parts['x'], bins=100)
        ax1.set_xlabel('x [mm]')
        ax1.set_ylabel("")
        #ax1.set_yscale("log")
        ax1.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e3:.{1}f}" for x in plt.xticks()[0]])


        ax2 = fig1.add_subplot(2,3,2)
        ax2.hist(cry_impact_parts['y'], bins=100) 
        ax2.set_xlabel('y [mm]')
        ax2.set_ylabel('')
        #ax2.set_yscale("log")
        ax2.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e3:.{1}f}" for x in plt.xticks()[0]])
        ax2.set_title(f'Total particles hitting the crystal: {len(cry_impact_parts)}')

        ax3 = fig1.add_subplot(2,3,3)
        h = ax3.hist2d(cry_impact_parts['x'], cry_impact_parts['y'], bins=100, norm=matplotlib.colors.LogNorm())#, range = ([-40e-6, 40e-6], [-40e-6,40e-6])) 
        ax3.set_xlabel(r'x [mm]')
        ax3.set_ylabel(r'y [mm]')
        ax3.set_ylim(0,0.008)
        ax3.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e3:.{1}f}" for x in plt.xticks()[0]])
        ax3.set_yticks(ticks=plt.yticks()[0], labels=[f"{y*1e3:.{1}f}" for y in plt.yticks()[0]])
        
        axins = inset_axes(ax3, height="100%",  width="5%", loc='right', borderpad=-6 )
        fig1.colorbar(h[3], cax=axins, orientation='vertical', label='Count (log scale)')

        ax3_tw = ax3.twinx()
        ticks = np.arange(1, max(ax3.get_yticks())/sigma+1, 2.0)
    
        ax3_tw.set_yticks(ticks)
        #ax3_tw.set_ybound(ax3.get_ybound())
        ax3_tw.set_ylabel(r' n $\sigma$')
        #ax3_tw.set_yticklabels([f"{x /sigma :.{0}f}"  for x in ticks])
        ax3_tw.axhline(5, color = 'r', linestyle = '--')
        ax3_tw.text( max(ax3.get_xticks())-1.5e-3, 4, r'TCP $\sigma$')
        ax3.grid(linestyle=':')


        ax12 = fig1.add_subplot(2,3,4)
        ax12.hist(cry_impact_parts['px'], bins=100)
        ax12.set_xlabel(r'px [$\mu$rad]')
        ax12.set_ylabel("")
        ax12.set_yscale("log")
        ax12.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e6:.{0}f}" for x in plt.xticks()[0]])
        #ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)


        ax22 = fig1.add_subplot(2,3,5)
        ax22.hist(cry_impact_parts['py'], bins=100) 
        ax22.set_xlabel(r'py [$\mu$rad]')
        ax22.set_ylabel('')
        ax22.set_yscale("log")
        ax22.axvline(py_central, color = 'red', linestyle = '-', alpha = 0.8)
        ax22.axvline(py_central + critical_angle, color = 'red', linestyle = '--', alpha = 0.9)
        ax22.axvline(py_central - critical_angle, color = 'red', linestyle = '--', alpha = 0.9)
        ax22.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e6:.{0}f}" for x in plt.xticks()[0]])
        #ax22.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        chann = len(cry_impact_parts[(cry_impact_parts.py > py_central - critical_angle) & (cry_impact_parts.py < py_central + critical_angle)])
        ax22.set_title(f'N particle inside critical angle range: {chann}')

        ax32 = fig1.add_subplot(2,3,6)
        h2 = ax32.hist2d(cry_impact_parts['px'], cry_impact_parts['py'], bins=100, norm=matplotlib.colors.LogNorm(), range = ([-40e-6, 40e-6], [-40e-6,40e-6])) 
        ax32.set_xlabel(r'px [$\mu$rad]')
        ax32.set_ylabel(r'py [$\mu$rad]')
        ax32.set_xticks(ticks=plt.xticks()[0], labels=[f"{x*1e6:.{0}f}" for x in plt.xticks()[0]])
        ax32.set_yticks(ticks=plt.yticks()[0], labels=[f"{y*1e6:.{0}f}" for y in plt.yticks()[0]])
        axins_2 = inset_axes(ax32, height="100%",  width="5%", loc='right', borderpad=-6 )
        fig1.colorbar(h2[3], cax=axins_2, orientation='vertical', label='Count (log scale)')
        ax32.grid(linestyle=':')

        fig1.suptitle('CRY1 at ' + n_sig + r'$\sigma$')


        fig1.savefig("./Outputdata/impact_angles_" + n_sig + ".png")




#   ----------------------------------   MAIN     --------------------------------------------------------------------------

def main():

    if sys.argv[1] == '--ang_distr':
        prefix_name = sys.argv[2] if len(sys.argv) > 2 else 'TEST_B2V_ABS_CRY1_5.5'
        config_file = sys.argv[3] if len(sys.argv) > 3 else 'config_sim.yaml'
        angular_distribution(prefix_name, config_file)

    else:
        raise ValueError('The mode must be one of --, --')

if __name__ == '__main__':

    main()