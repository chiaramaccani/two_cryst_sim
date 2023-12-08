import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import yaml
import h5py
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xpart as xp
import xcoll as xc

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








def main():

    n_part = 10000000

    # EXAMPLE

    """x = np.random.uniform(-0.001, 0.001, n_part)
    px = np.zeros(n_part)
    y = np.random.uniform(0.0, 0.002, n_part)
    py = np.random.normal(0.0, 1e-6, n_part)

    crystal = SimpleCrystal(align_angle=0.0, jaw_L=0.0)"""

    p0c_ft = 6.8e12



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



    #align_angle_TCCS_5s = 0 
    x = np.random.uniform(-0.001, 0.001, n_part)
    px = np.zeros(n_part)
    y = np.random.uniform(0.0 + jaw_L_TCCS_5s, ydim_TCCS + jaw_L_TCCS_5s, n_part)
    py = np.random.normal(align_angle_TCCS_5s, 1e-6, n_part)

    #theta_c_Si = 0.8e-6
    #print(theta_c_Si)



    part = xp.Particles(x=x, 
                        px=px, 
                        y=y, 
                        py=py, 
                        p0c=p0c_ft)
    

    
    crystal = SimpleCrystal(
                        align_angle = align_angle_TCCS_5s, 
                        jaw_L = jaw_L_TCCS_5s,
                        length = length_TCCS, 
                        xdim = xdim_TCCS, 
                        ydim = ydim_TCCS,
                        bend = bend_TCCS, 
                        theta_c_Si = theta_c_Si, 
                        )
                        
    drift_length = 200
    drift =  xt.Drift(length=drift_length)

    line = xt.Line(elements=[crystal, drift], element_names=["TCCS", "drift"])
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=6))

    line.track(part, num_turns=1)

    fig1 = plt.figure(figsize=(19, 12)) #figsize=(24, 5)
    ax1 = fig1.add_subplot(2,2,1)
    ax1.hist(y, bins=1000, histtype='step', density=True, range=[-4e-3, ydim_TCCS + jaw_L_TCCS_5s + bend_angle_TCCS * drift_length ], label = 'incoming')
    ax1.hist(part.y, bins=1000, histtype='step', density=True, range=[-4e-3, ydim_TCCS + jaw_L_TCCS_5s  + bend_angle_TCCS * drift_length ], label = 'outgoing')
    ax1.legend()
    ax1.set_xlabel('y [m]')
  
    ax2 = fig1.add_subplot(2,2,2)
    ax2.hist(py, bins=1000, histtype='step', density=True,  label = 'incoming')#, range=[align_angle_TCCS_5s - 1e-5, align_angle_TCCS_5s + bend_angle_TCCS + 1e-5])
    ax2.hist(part.py, bins=1000, histtype='step', density=True, label = 'outgoing')#, #range=[align_angle_TCCS_5s - 1e-5, align_angle_TCCS_5s + bend_angle_TCCS + 1e-5]) #part.py-py
    ax2.set_xlabel('py [rad]')
    ax2.legend()

    ax3 = fig1.add_subplot(2,2,3)
    ax3.hist2d(py, part.py, bins=500, range=[[align_angle_TCCS_5s - 1e-5, align_angle_TCCS_5s + 1e-5], [align_angle_TCCS_5s - 1e-5, align_angle_TCCS_5s + bend_angle_TCCS + 1e-5]]) 
    ax3.set_xlabel('y [m]')
    ax3.set_xlabel('py [rad]')

    ax4 = fig1.add_subplot(2,2,4)
    ax4.hist(py, bins=1000, histtype='step', density=True, label = 'incoming')#, range=[align_angle_TCCS_5s - 1e-5, align_angle_TCCS_5s + bend_angle_TCCS + 1e-5])
    ax4.hist(part.py-py, bins=1000, histtype='step', density=True , label= 'outgoing-incoming')
    ax4.set_xlabel('py [rad]')
    ax4.legend()

    plt.savefig("./Outputdata/Bjorn_sim_TCCS.png")

if __name__ == "__main__":
    main()