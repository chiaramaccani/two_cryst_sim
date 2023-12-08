import json
import numpy as np
from pathlib import Path
import sys
import os
import yaml

import xobjects as xo
import xtrack as xt
import xpart as xp
import xcoll as xc
import pickle 


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




# --------------------------------  SIMPLE MODEL -------------------------------------------------------------

class SimpleCrystal(xt.BeamElement):
    _xofields = {
        "length": xo.Float64,
        "xdim": xo.Float64, 
        "ydim": xo.Float64,  
        "jaw_L": xo.Float64, 
        "align_angle": xo.Float64, 
        "dp_Si": xo.Float64,
        "theta_c_Si": xo.Float64, 
        "bend": xo.Float64, 
        "fit_coeffs": xo.Float64[:], 
        "len_fit_coeffs": xo.Int64, 
    }

    isthick = False
    behaves_like_drift = False

    _extra_c_sources = [
        '''
        #ifndef XTRACK_SIMPLECRYSTAL_H
        #define XTRACK_SIMPLECRYSTAL_H

        #include <math.h>
        //#include <stdio.h>

        /*gpufun*/
        void SimpleCrystal_track_local_particle(SimpleCrystalData el, LocalParticle* part0){
            

            double length = SimpleCrystalData_get_length(el);
            double xdim = SimpleCrystalData_get_xdim(el);
            double ydim = SimpleCrystalData_get_ydim(el);
            double jaw_L = SimpleCrystalData_get_jaw_L(el);
            double align_angle = SimpleCrystalData_get_align_angle(el);
            double dp_Si = SimpleCrystalData_get_dp_Si(el);
            double theta_c_Si = SimpleCrystalData_get_theta_c_Si(el);
            double bend = SimpleCrystalData_get_bend(el);

            double theta_bend = length / bend;

            int len_fit_coeffs = SimpleCrystalData_get_len_fit_coeffs(el);

            //start_per_particle_block (part0->part)

                double x = LocalParticle_get_x(part);
                double y = LocalParticle_get_y(part);
                double py = LocalParticle_get_py(part);

                if ( (x <= xdim/2.0) && (x >= -xdim/2.0) && (y >= jaw_L) && (y <= jaw_L + ydim)&& (py <= align_angle + theta_c_Si) && (py >= align_angle -theta_c_Si)) {  
                    
                    //printf("%lf", align_angle);
                    double y_in = -1 + 2 * ((float)rand()) / RAND_MAX;
                    //double yp_in = py / theta_c_Si;
                    double yp_in = (py - align_angle) / theta_c_Si;


                    if (yp_in * yp_in / (1 - y_in * y_in) < 1) {
                    
                        double ph_in = atan2(yp_in, y_in);
                        double A = sqrt(y_in * y_in + yp_in * yp_in);

                        double lambda = 0.0;
                        for (int i = 0; i < len_fit_coeffs; i++) {
                            double A_pow = 1.0;
                            for (int j = 0; j < (len_fit_coeffs - i - 1); j++) {
                                A_pow *= A;
                            }
                            lambda += SimpleCrystalData_get_fit_coeffs(el, i) * A_pow;
                        }

                        double phAdv = 2*M_PI*1e6/lambda;

                        double y_out = A * sin(ph_in + length * phAdv) * dp_Si/2;
                        double yp_out = A * cos(ph_in + length * phAdv) * theta_c_Si;   

                        LocalParticle_set_y(part, y_out + y + length * theta_bend * 0.5);
                        LocalParticle_set_py(part, yp_out + align_angle + theta_bend);
                    }
                }

            //end_per_particle_block

        }

        #endif /* XTRACK_SIMPLECRYSTAL_H */
        '''
    ]

    def __init__(self, length = 0.002, 
                       xdim = 0.05, 
                       ydim = 0.002,
                       jaw_L = 0.0015, 
                       align_angle = 12e-6, 
                       bend = 40, 
                       theta_c_Si = 1.5e-6, 
                       **kwargs):
        
        wavelength = np.array([[-0.784,0.841],[-0.768,0.746],[-0.749,0.673],[-0.735,0.649],[-0.703,0.63],[-0.68,0.638],
                        [-0.652,0.655],[-0.634,0.676],[-0.606,0.695],[-0.578,0.732],[-0.536,0.78],[-0.495,0.803],
                        [-0.444,0.826],[-0.375,0.851],[-0.324,0.88],[-0.268,0.913],[-0.222,0.947],[-0.19,0.97],
                        [-0.166,0.991],[-0.111,0.977],[-0.069,0.965],[0,0.961],[0.069,0.964],[0.12,0.976],[0.171,0.991],
                        [0.217,0.961],[0.264,0.925],[0.314,0.895],[0.379,0.859],[0.449,0.827],[0.509,0.8],[0.541,0.784],
                        [0.573,0.748],[0.61,0.706],[0.647,0.667],[0.684,0.639],[0.712,0.63],[0.735,0.642],[0.763,0.671],
                        [0.772,0.714],[0.782,0.762],[0.793,0.838]])
        dp_Si = 1.92e-10
        dp_W = 1.58e-10
        theta_c_W = 343e-6
        wavelength2 = wavelength.copy()
        t1 = wavelength2[:21,0]*1/np.abs(wavelength2[0][0])
        t2 = wavelength2[21:,0]*1/np.abs(wavelength2[-1][0])
        t0 = np.append(t1,t2)
        wavelength2[:,1] = wavelength2[:,1]*2*(dp_Si/dp_W)*(theta_c_W/theta_c_Si) ## scale to our crystal, and lambda/2 -> lambda
        wavelength2[:,0] = t0

        lambdaFit2=np.polyfit(wavelength2[:,0],wavelength2[:,1],30)
        
        if '_xobject' not in kwargs:
            kwargs.setdefault('length', length)
            kwargs.setdefault('xdim', xdim)
            kwargs.setdefault('ydim', ydim)
            kwargs.setdefault('jaw_L', jaw_L)
            kwargs.setdefault('align_angle', align_angle)
            kwargs.setdefault('dp_Si', 1.92e-10)
            kwargs.setdefault('bend', bend)
            kwargs.setdefault('theta_c_Si', theta_c_Si)
            kwargs.setdefault("fit_coeffs", lambdaFit2)
            kwargs.setdefault("len_fit_coeffs", len(lambdaFit2))

        super().__init__(**kwargs)

    has_backtrack = False

# --------------------------------  SIMPLE MODEL -------------------------------------------------------------



def main():


    
    config_file = sys.argv[1]
    
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    run_dict = config_dict['run']
    file_dict = config_dict['input_files']

    mode = run_dict['mode']


    if mode == 'xsuite':
        coll_file = config_dict['input_files']['collimators']
    elif mode == 'simple_model':
        coll_file = '/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim/input_files/flat_top_bjorn.yaml'

    with open(coll_file, 'r') as stream:
        coll_dict = yaml.safe_load(stream)['collimators']['b'+config_dict['run']['beam']]


    context = xo.ContextCpu(omp_num_threads='auto')

    # On a modern CPU, we get ~5000 particle*turns/s
    # So this script should take around half an hour
    beam          = run_dict['beam']
    plane         = run_dict['plane']

    num_turns     = run_dict['turns']
    num_particles = run_dict['nparticles']
    engine        = run_dict['engine']

    TTCS_align_angle_step = run_dict['TTCS_align_angle_step']



    path_out = Path.cwd() / 'Outputdata'

    if not path_out.exists():
        os.makedirs(path_out)


    # Load from json
    line = xt.Line.from_json(file_dict[f'line_b{beam}'])
    line.particle_ref = xp.Particles(p0c=1e12, q0=1, mass0=xp.PROTON_MASS_EV)
    print(line.particle_ref)
    

    end_s = line.get_length()

    TCCS_name = 'tccs.5r3.b2'
    TCCP_name = 'tccp.4l3.b2'
    TARGET_name = 'target.4l3.b2'
    TCLA_name = 'tcla.a5l3.b2'

    TCCS_loc = end_s - 6773.7 #6775
    TCCP_loc = end_s - 6653.3 #6655
    TARGET_loc = end_s - (6653.3 +  0.070/2 + coll_dict[TARGET_name]["length"]/2)
    TCLA_loc = line.get_s_position()[line.element_names.index(TCLA_name)]


    if(mode == 'xsuite'):
        line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)
        line.insert_element(at_s=TCCS_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCS_name+'_aper')
        line.insert_element(at_s=TARGET_loc, element=xt.Marker(), name=TARGET_name)
        line.insert_element(at_s=TARGET_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TARGET_name+'_aper')
        line.insert_element(at_s=TCCP_loc, element=xt.Marker(), name=TCCP_name)
        line.insert_element(at_s=TCCP_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCP_name+'_aper')

        print(coll_dict.keys())

    elif(mode == 'simple_model'):

        p0c_ft = 1e12

        #CRY1 at 5 sigma
        align_angle_TCCS_5s = -1.1763616021881982e-05   # align_angle = 12e-6, 
        jaw_L_TCCS_5s = 0.0016912979598174786           # jaw_L = 0.0015, 
        length_TCCS = 0.004                             # length = 0.002, 
        bend_TCCS = 80                                  # bend = 40, 
        bend_angle_TCCS = length_TCCS / bend_TCCS
        # invert x and y dimension, angle: 90 
        xdim_TCCS = 0.035                               # xdim = 0.05, 
        ydim_TCCS = 0.002                               # ydim = 0.002,
        
        pot_crit_Si = 21.34 #16 #eV
        en_crit_Si = 5.7e9 / 1e-2 #eV/m
        dp_Si = 1.92e-10 #m

        xp_crit0 = np.sqrt(2.0*pot_crit_Si/p0c_ft)
        Rcrit = p0c_ft/en_crit_Si
        theta_c_Si = xp_crit0*(1-Rcrit/bend_TCCS)
   
        crystal_TCCS = SimpleCrystal(
                            align_angle = align_angle_TCCS_5s, 
                            jaw_L = jaw_L_TCCS_5s,
                            length = length_TCCS, 
                            xdim = xdim_TCCS, 
                            ydim = ydim_TCCS,
                            bend = bend_TCCS, 
                            theta_c_Si = theta_c_Si, 
                            )
        
        line.insert_element(at_s=TCCS_loc, element=crystal_TCCS, name=TCCS_name)
        line.insert_element(at_s=TCCS_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TCCS_name+'_aper')
        line.insert_element(at_s=TARGET_loc, element=xt.Marker(), name=TARGET_name)
        line.insert_element(at_s=TARGET_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=TARGET_name+'_aper')



    TCCS_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
    TARGET_monitor = xt.ParticlesMonitor(num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns)
    dx = 1e-11
    line.insert_element(at_s = TCCS_loc - 0.004/2 - dx, element=TCCS_monitor, name='TCCS_monitor')
    line.insert_element(at_s = TARGET_loc - coll_dict[TARGET_name]["length"]/2 - dx, element=TARGET_monitor, name='TARGET_monitor')


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
    coll_manager = xc.CollimatorManager.from_yaml(coll_file, line=line, beam=beam, _context=context, ignore_crystals=False)

    #print(coll_manager.collimator_names)

    # Install collimators into line
    if engine == 'everest':
        coll_names = coll_manager.collimator_names
        black_absorbers = [TARGET_name,]
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


    # Aperture model check
    print('\nAperture model check after introducing collimators:')
    df_with_coll = line.check_aperture()
    assert not np.any(df_with_coll.has_aperture_problem)


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
    line.track(part, num_turns=num_turns, time=True)
    coll_manager.disable_scattering()
    print(f"Done tracking in {line.time_last_track:.1f}s.")


    # Save lossmap to json, which can be loaded, combined (for more statistics),
    # and plotted with the 'lossmaps' package
    _ = coll_manager.lossmap(part, file=Path(path_out,f'lossmap_B{beam}{plane}.json'))


    # Save a summary of the collimator losses to a text file
    summary = coll_manager.summary(part) #, file=Path(path_out,f'coll_summary_B{beam}{plane}.out')
    print(summary)

    TCCS_monitor_dict = TCCS_monitor.to_dict()
    TARGET_monitor_dict = TARGET_monitor.to_dict()
    with open(Path(path_out,f'TCCS_monitor_B{beam}{plane}_{mode}.pkl'), 'wb') as f:
        pickle.dump(TCCS_monitor_dict, f)
    with open(Path(path_out,f'TARGET_monitor_B{beam}{plane}_{mode}.pkl'), 'wb') as f:
        pickle.dump(TARGET_monitor_dict, f)


if __name__ == "__main__":
    main()

