import json
import os
import subprocess

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

import json

import xtrack as xt
import xpart as xp
import xobjects as xo


import pickle 
import h5py
import io
import scipy



class LineData():
    
    def __init__(self, run, # 'HL' or 'Run3'
                 line_file_name, # = 'config_sim.yaml',
                 coll_file_name, # = 'config_sim.yaml'
                 TCCS_name = 'tccs.5r3.b2',
                 TCCP_name = 'tccp.4l3.b2',
                 TARGET_name = 'target.4l3.b2',
                 TCCS_loc_b1 =  6773.7,
                 TCCP_loc_b1 = 6653.3,
                 beam = 2, plane = 'V', engine = 'everest', sigma_TCCS = None, sigma_TCCP = None,
                 job_num_part = 100000, job_num_turns = 200, 
              ):

        self.run = run
        self.line_file = f"{os.environ.get('HOME_TWOCRYST')}/{line_file_name}"
        self.coll_file = f"{os.environ.get('HOME_TWOCRYST')}/{coll_file_name}"
        
        self.TCCS_name = TCCS_name
        self.TCCP_name = TCCP_name
        self.TARGET_name = TARGET_name
        self.TCLA_name = 'tcla.a5l3.b2'
        self.TCCS_loc_b1 =  6773.7
        self.TCCP_loc_b1 = 6653.3
        self.TCCS_loc = None
        self.TCCP_loc = None
        self.TARGET_loc = None
        self.TCLA_loc = None
        self.idx_TCCS = None
        self.idx_TCCP = None
        self.idx_TARGET = None
        
        self.beam = beam
        self.plane = plane
        self.job_num_part = job_num_part
        self.job_num_turns = job_num_turns
        self.engine = engine
        self.plane = plane
        self.sigma_TCCS = sigma_TCCS
        self.sigma_TCCP = sigma_TCCP
        self.coll_dict = None
        self.end_s = None
        self.line = None
        self.norm_emittance = None
        self.emittance = None
        
        if self.run == 'Run3':
            self.norm_emittance = 3.5e-6
        if self.run == 'HL':
            self.norm_emittance = 2.5e-6
        
    def load_colldb_new(self, filename):
        with open(filename, "r") as infile:
            coll_data_string = ""
            family_settings = {}
            family_types = {}
            onesided = {}
            tilted = {}
            bend = {}
            xdim = {}
            ydim = {}

            for l_no, line in enumerate(infile):
                if line.startswith("#"):
                    continue  # Comment
                if len(line.strip()) == 0:
                    continue  # Empty line
                sline = line.split()
                if len(sline) < 6 or sline[0].lower() == "crystal" or sline[0].lower() == "target":
                    if sline[0].lower() == "nsig_fam":
                        family_settings[sline[1]] = sline[2]
                        family_types[sline[1]] = sline[3]
                    elif sline[0].lower() == "onesided":
                        onesided[sline[1]] = int(sline[2])
                    elif sline[0].lower() == "tilted":
                        tilted[sline[1]] = [float(sline[2]), float(sline[3])]
                    elif sline[0].lower() == "crystal":
                        bend[sline[1]] = float(sline[2])
                        xdim[sline[1]] = float(sline[3])
                        ydim[sline[1]] = float(sline[4])
                    elif sline[0].lower() == "target":
                        xdim[sline[1]] = float(sline[2])
                        ydim[sline[1]] = float(sline[3])
                    elif sline[0].lower() == "settings":
                        pass  # Acknowledge and ignore this line
                    else:
                        raise ValueError(f"Unknown setting {line}")
                else:
                    coll_data_string += line

        names = ["name", "opening", "material", "length", "angle", "offset"]

        df = pd.read_csv(io.StringIO(coll_data_string), delim_whitespace=True,
                         index_col=False, skip_blank_lines=True, names=names)

        df["angle"] = df["angle"] 
        df["name"] = df["name"].str.lower() # Make the names lowercase for easy processing
        df["gap"] = df["opening"].apply(lambda s: float(family_settings.get(s, s)))
        df["type"] = df["opening"].apply(lambda s: family_types.get(s, "UNKNOWN"))
        df["side"] = df["name"].apply(lambda s: onesided.get(s, 0))
        df["bend"] = df["name"].apply(lambda s: bend.get(s, 0))
        df["xdim"] = df["name"].apply(lambda s: xdim.get(s, 0))
        df["ydim"] = df["name"].apply(lambda s: ydim.get(s, 0))
        df["tilt_left"] = df["name"].apply(lambda s: np.deg2rad(tilted.get(s, [0, 0])[0]))
        df["tilt_right"] = df["name"].apply(lambda s: np.deg2rad(tilted.get(s, [0, 0])[1]))
        df.rename(columns={"opening": "family"}, inplace=True)
        df = df.set_index("name").T

        # Ensure the collimators marked as one-sided or tilted are actually defined
        defined_set = set(df.columns) # The data fram was transposed so columns are names
        onesided_set = set(onesided.keys())
        tilted_set = set(tilted.keys())
        if not onesided_set.issubset(defined_set):
            different = onesided_set - defined_set
            raise SystemExit('One-sided collimators not defined: {}'.format(", ".join(different)))
        if not tilted_set.issubset(defined_set):
            different = tilted_set - defined_set
            raise SystemExit('Tilted collimators not defined: {}'.format(",".join(different)))
        return df.T


    def find_axis_intercepts(self, x_coords, y_coords):
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



    def find_bad_offset_apertures(self, line):
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
                x_intercepts, y_intercepts = self.find_axis_intercepts(ap_dict['x_vertices'],
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

        
    def load_line(self):
        
        #TTCS_align_angle_step = run_dict['TTCS_align_angle_step']

        #mode = run_dict['mode']
        #print('\nMode: ', mode, '\n')

        print('Input files:\n', self.line_file, '\n', self.coll_file, '\n')

        if self.coll_file.endswith('.yaml'):
            with open(self.coll_file, 'r') as stream:
                coll_dict = yaml.safe_load(stream)['collimators'][f'b{self.beam}']
        if self.coll_file.endswith('.data'):
            coll_dict = self.load_colldb_new(self.coll_file).to_dict('index')

        context = xo.ContextCpu(omp_num_threads='auto')
        self.coll_dict = coll_dict
        
        # Load Line in Xtrack
        line = xt.Line.from_json(self.line_file)
        self.end_s = line.get_length()
        
        self.TCCS_loc = self.end_s - self.TCCS_loc_b1 #6775
        self.TCCP_loc = self.end_s - self.TCCP_loc_b1 #6655
        self.TARGET_loc = self.end_s - (self.TCCP_loc_b1 + coll_dict[self.TCCP_name]["length"]/2 + coll_dict[self.TARGET_name]["length"]/2)
        self.TCLA_loc = line.get_s_position()[line.element_names.index(self.TCLA_name)]

        line.insert_element(at_s=self.TCCS_loc, element=xt.Marker(), name=self.TCCS_name)
        line.insert_element(at_s=self.TCCS_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=self.TCCS_name+'_aper')
        line.insert_element(at_s=self.TARGET_loc, element=xt.Marker(), name=self.TARGET_name)
        line.insert_element(at_s=self.TARGET_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=self.TARGET_name+'_aper')
        line.insert_element(at_s=self.TCCP_loc, element=xt.Marker(), name=self.TCCP_name)
        line.insert_element(at_s=self.TCCP_loc, element=xt.LimitEllipse(a_squ=0.0016, b_squ=0.0016, a_b_squ=2.56e-06), name=self.TCCP_name+'_aper')

        TCCS_monitor = xt.ParticlesMonitor(num_particles=self.job_num_part, start_at_turn=0, stop_at_turn=self.job_num_turns)
        TARGET_monitor = xt.ParticlesMonitor(num_particles=self.job_num_part, start_at_turn=0, stop_at_turn=self.job_num_turns)
        dx = 1e-11
        line.insert_element(at_s = self.TCCS_loc - coll_dict[self.TCCS_name]["length"]/2 - dx, element=TCCS_monitor, name='TCCS_monitor')
        line.insert_element(at_s = self.TARGET_loc - coll_dict[self.TARGET_name]["length"]/2 - dx, element=TARGET_monitor, name='TARGET_monitor')


        bad_aper = self.find_bad_offset_apertures(line)
        print('Bad apertures : ', bad_aper)
        print('Replace bad apertures with Marker')
        for name in bad_aper.keys():
            line.element_dict[name] = xt.Marker()
            print(name, line.get_s_position(name), line.element_dict[name])

        """# Aperture model check
        print('\nAperture model check on imported model:')
        df_imported = line.check_aperture()
        assert not np.any(df_imported.has_aperture_problem)"""


        # Initialise collmanager
        if self.coll_file.endswith('.yaml'):
            coll_manager = xc.CollimatorManager.from_yaml(self.coll_file, line=line, beam=self.beam, _context=context, ignore_crystals=False)
        elif self.coll_file.endswith('.data'):
            coll_manager = xc.CollimatorManager.from_SixTrack(self.coll_file, line=line, _context=context, ignore_crystals=False, nemitt_x = 2.5e-6,  nemitt_y = 2.5e-6)
            # switch on cavities
            speed = line.particle_ref._xobject.beta0[0]*scipy.constants.c
            harmonic_number = 35640
            voltage = 12e6/len(line.get_elements_of_type(xt.Cavity)[1])
            frequency = harmonic_number * speed /line.get_length()
            for side in ['l', 'r']:
                for cell in ['a','b','c','d']:
                    line[f'acsca.{cell}5{side}4.b2'].voltage = voltage
                    line[f'acsca.{cell}5{side}4.b2'].frequency = frequency

        # Install collimators into line
        if self.engine == 'everest':
            coll_names = coll_manager.collimator_names
            black_absorbers = [self.TARGET_name,]

            everest_colls = [name for name in coll_names if name not in black_absorbers]
            coll_manager.install_everest_collimators(names=everest_colls,verbose=True)
            coll_manager.install_black_absorbers(names = black_absorbers, verbose=True)
        else:
            raise ValueError(f"Unknown scattering engine {self.engine}!")
        
        """# Aperture model check
        print('\nAperture model check after introducing collimators:')
        df_with_coll = line.check_aperture()
        assert not np.any(df_with_coll.has_aperture_problem)"""
      
        # Build the tracker
        coll_manager.build_tracker()

        # Set the collimator openings based on the colldb,
        # or manually override with the option gaps={collname: gap}
        coll_manager.set_openings()

        """# Aperture model check
        print('\nAperture model check after introducing collimators:')
        df_with_coll = line.check_aperture()
        assert not np.any(df_with_coll.has_aperture_problem)"""
        
        self.line = line         
        return
    
    def compute_sigma_element(self, element_name):
        if self.line is None:
            self.load_line()
        twiss = self.line.twiss()
        beta_y_optics = twiss['bety', element_name]
        alfa_y_optics = twiss['alfy', element_name]
        if self.emittance is None:
            self.emittance_phy = self.norm_emittance/(self.line.particle_ref._xobject.beta0[0]*self.line.particle_ref._xobject.gamma0[0])
        sigma = np.sqrt(self.emittance_phy*beta_y_optics)
        return(sigma)
    
    def compute_crystals(self):
        if self.line is None:
            self.load_line()
        self.idx_TCCS = self.line.element_names.index(self.TCCS_name)
        self.idx_TARGET = self.line.element_names.index(self.TARGET_name)
        self.idx_TCCP = self.line.element_names.index(self.TCCP_name)
        
        print(f"\n\nParticleAnalysis(element_type=\'crystal\', n_sigma={self.coll_dict[self.TCCS_name]['gap']}, length={self.coll_dict[self.TCCS_name]['length']}, ydim={self.coll_dict[self.TCCS_name]['xdim']}, xdim={self.coll_dict[self.TCCS_name]['ydim']}, bend={self.coll_dict[self.TCCS_name]['bend']}, align_angle={self.line.elements[self.idx_TCCS].align_angle}, jaw_L={self.line.elements[self.idx_TCCS].jaw_L}), line_idx={self.idx_TCCS}")
        print(f"ParticleAnalysis(element_type=\'target\', n_sigma={self.coll_dict[self.TARGET_name]['gap']}, length={self.coll_dict[self.TARGET_name]['length']}, ydim={self.coll_dict[self.TARGET_name]['xdim']}, xdim={self.coll_dict[self.TARGET_name]['ydim']}, jaw_L={self.line.elements[self.idx_TARGET].jaw_L}), line_idx={self.idx_TARGET}")
        print(f"ParticleAnalysis(element_type=\'crystal\', n_sigma={self.coll_dict[self.TCCP_name]['gap']}, length={self.coll_dict[self.TCCP_name]['length']}, ydim={self.coll_dict[self.TCCP_name]['xdim']}, xdim={self.coll_dict[self.TCCP_name]['ydim']}, bend={self.coll_dict[self.TCCP_name]['bend']}, jaw_L={self.line.elements[self.idx_TCCP].jaw_L}), line_idx={self.idx_TCCP}")
        
        
        
        
line_test0 = LineData(run='HL', line_file_name = 'input_files/HL_IR7_rematched/b4_sequence_patched.json', 
                     coll_file_name = 'input_files/CollDB_HL_tight_b4.data')
line_test1 = LineData(run='HL', line_file_name = 'input_files/HL_IR7_IR3_rematched/b4_sequence_patched.json', 
                     coll_file_name = 'input_files/CollDB_HL_tight_b4.data')
line_test2 = LineData(run='Run3', line_file_name = 'input_files/flat_top_b2.json', 
                     coll_file_name = 'input_files/flat_top.yaml')

line_test0.compute_crystals()
line_test1.compute_crystals()
line_test2.compute_crystals()
