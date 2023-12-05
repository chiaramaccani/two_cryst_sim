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
        "lcry": xo.Float64,
        "xdim": xo.Float64, 
        "ydim": xo.Float64,  
        "jaw": xo.Float64, 
        "align_angle": xo.Float64, 
        "dp_Si": xo.Float64,
        "theta_c_Si": xo.Float64, 
        "theta_bend": xo.Float64, 
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

        /*gpufun*/
        void SimpleCrystal_track_local_particle(SimpleCrystalData el, LocalParticle* part0){

            double lcry = SimpleCrystalData_get_lcry(el);
            double xdim = SimpleCrystalData_get_xdim(el);
            double ydim = SimpleCrystalData_get_ydim(el);
            double jaw = SimpleCrystalData_get_jaw(el);
            double align_angle = SimpleCrystalData_get_align_angle(el);
            double dp_Si = SimpleCrystalData_get_dp_Si(el);
            double theta_c_Si = SimpleCrystalData_get_theta_c_Si(el);
            double theta_bend = SimpleCrystalData_get_theta_bend(el);

            int len_fit_coeffs = SimpleCrystalData_get_len_fit_coeffs(el);

            //start_per_particle_block (part0->part)

                double x = LocalParticle_get_x(part);
                double y = LocalParticle_get_y(part);
                double py = LocalParticle_get_py(part);

                if ((x <= xdim/2.0) && (x >= -xdim/2.0) && (y >= jaw) && (y <= jaw + ydim) && (py <= align_angle + theta_c_Si) && (py >= align_angle -theta_c_Si)) {
                    double x_in = -1 + 2 * ((float)rand()) / RAND_MAX;
                    double xp_in = py / theta_c_Si;

                    if (xp_in * xp_in / (1 - x_in * x_in) < 1) {
                        double ph_in = atan2(xp_in, x_in);
                        double A = sqrt(x_in * x_in + xp_in * xp_in);

                        double lambda = 0.0;
                        for (int i = 0; i < len_fit_coeffs; i++) {
                            double A_pow = 1.0;
                            for (int j = 0; j < (len_fit_coeffs - i - 1); j++) {
                                A_pow *= A;
                            }
                            lambda += SimpleCrystalData_get_fit_coeffs(el, i) * A_pow;
                        }

                        double phAdv = 1e6/lambda;

                        double x_out = A * sin(ph_in + lcry * phAdv) * dp_Si;
                        double xp_out = A * cos(ph_in + lcry * phAdv) * theta_c_Si;   

                        LocalParticle_set_y(part, x_out + y + lcry * theta_bend*0.5);
                        LocalParticle_set_py(part, xp_out + align_angle + theta_bend);
                    }
                }

            //end_per_particle_block

        }

        #endif /* XTRACK_SIMPLECRYSTAL_H */
        '''
    ]

    def __init__(self, lcry = 0.002, 
                       xdim = 0.05, 
                       ydim = 0.002,
                       jaw = 0.0015, 
                       align_angle = 12e-6, 
                       theta_bend = 50e-6, 
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
            kwargs.setdefault('lcry', lcry)
            kwargs.setdefault('xdim', xdim)
            kwargs.setdefault('ydim', ydim)
            kwargs.setdefault('jaw', jaw)
            kwargs.setdefault('align_angle', align_angle)
            kwargs.setdefault('dp_Si', 1.92e-10)
            kwargs.setdefault('theta_bend', theta_bend)
            kwargs.setdefault('theta_c_Si', theta_c_Si)
            kwargs.setdefault("fit_coeffs", lambdaFit2)
            kwargs.setdefault("len_fit_coeffs", len(lambdaFit2))

        super().__init__(**kwargs)

    has_backtrack = False


def bs():
    ### halv wavelength depending on incoming x position within the channel for U238, 300 MeV/u, 
    ###  W crystal, dp = 1.58 Å, theta_c = 343 µrad,
    ###  lamda ~= pi*dp/theta_c,
    ###  so we need to scale this by dp and theta_c
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
    theta_c_Si = XXX
    wavelength2 = wavelength.copy()
    t1 = wavelength2[:21,0]*1/np.abs(wavelength2[0][0])
    t2 = wavelength2[21:,0]*1/np.abs(wavelength2[-1][0])
    t0 = np.append(t1,t2)
    wavelength2[:,1] = wavelength2[:,1]*2*(dp_Si/dp_W)*(theta_c_W/theta_c_Si) ## scale to our crystal, and lambda/2 -> lambda
    wavelength2[:,0] = t0

    lambdaFit2=np.polyfit(wavelength2[:,0],wavelength2[:,1],30)
    def mylambda(func,xx):
        yy = np.zeros(len(xx))
        for i in np.arange(len(func)):
            yy += func[i]*xx**(len(func)-i-1)
        return yy    

    ######

    x_in = np.random.uniform(-1,1,1)
    xp_in = xp0/theta_c_Si

    ## check if accepted for channeling:
    xp**2/(1-x**2)<1

    ph_in = np.arctan2(xp_in,x_in)
    A = np.sqrt(x**2+xp**2)

    phAdv = 1/mylambda(lambdaFit2,A)
    x_out  = A*np.sin(ph_in + lcry*phAdv)*dp_Si
    xp_out = A*np.cos(ph_in + lcry*phAdv)*theta_c_Si   

    x1  = x_out + x0 + lcry*theta_bend*0.5 ### NOTE crosscheck this
    xp1 = xp_out + theta_bend   


def main():
  

    x = np.random.uniform(-0.001, 0.001, 1000000)
    px = np.zeros(1000000)
    y = np.random.uniform(0.0, 0.002, 1000000)
    py = np.random.normal(0.0, 1e-6, 1000000)

    part = xp.Particles(x=x, 
                        px=px, 
                        y=y, 
                        py=py, 
                        p0c=6.8e12)


    crystal = SimpleCrystal(align_angle=0.0, jaw=0.0)

    line = xt.Line(elements=[crystal], element_names=["crys"])
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=6))

    line.track(part, num_turns=1)

    plt.figure()
    plt.hist(py, bins=1000, histtype='step', density=True, range=[-10e-6, 80e-6])
    plt.hist(part.py, bins=1000, histtype='step', density=True, range=[-10e-6, 80e-6])
    plt.show()

    plt.figure()
    plt.hist2d(py, part.py, bins=500, range=[[-10e-6, 10e-6], [40e-6, 60e-6]])
    plt.show()


if __name__ == "__main__":
    main()