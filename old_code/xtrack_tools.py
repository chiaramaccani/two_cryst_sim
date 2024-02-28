## XTracking tools 
## Classes and methods for usage in xtrack
## P. Hermes
## 18.03.2022


# Load required packages
# from xml.sax.handler import all_properties
import numpy as np
import yaml, sys, json
import subprocess

xt_dev = False

if xt_dev:
    sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xtrack")
    sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xpart")
    sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xdeps")
    sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xobjects")
    sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xfields")

# sys.path("/afs/cern.ch/work/p/pahermes/private/xsuite/dev")

# sys.path.append('/afs/cern.ch/work/p/pahermes/private/xsuite')
import xobjects as xo
import xpart as xp
import pandas as pd
import xtrack as xt
import warnings
import sqlite3

from slice_input_distribution import slice_distribution
sys.path.append('/afs/cern.ch/user/p/pahermes/public/python_packages/')

# tools to calculate RMS and for bootstrapping
from PyBeamPhysics.statistics import get_rms, bootstrap

# Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')

# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__


########################################################################
## Tools to load settings 
########################################################################

def _format_settings(settings):
    """Loads the settings from a dictionary and converts to float or
       int where possible.  
    """
    ## Settings as a dictionary 
    d = settings
    # convert everything to integer or float if possible
    for key, val in d.items():
        if isinstance(val, dict):
            _format_settings(val)
        else:
            try:
                val = float(val)
                if float(val) == int(val):
                    val = int(val)
                d[key] = val
            except:
                pass
    return d

## Function to load the settings from a file 
def load_settings(filename = "settings.yml"):
    """Function to load the settings from a file
       Returns settings as a dictionary. If the filename is not specified explicitly
       the default filename 'settings.yml' is used.

        Parameters
        ----------
        filename : str, optional
            Filename of the yaml file containing the settings (default is 'settings.yml')
    """
    ## Load settings from file using yaml
    with open(filename, 'r') as stream:
        settings_raw = yaml.safe_load(stream)
    settings = _format_settings(settings_raw)

    # load the context 
    get_context(settings)

    ## Check if settings check is activated
    if settings['settings']['verify']:
        pf("load_settings", "Settings verification activated")
        # check if the settings are correct
        verify_settings(settings)
    else:
        pf("load_settings", "Settings verification not activated")

    # verbose mode 
    try: 
        if settings['verbose']:
            pf("load_settings", "Verbose mode activated")
        else:
            pf("load_settings", "Verbose mode not activated")
    except KeyError:
        pf("load_settings", "Verbose mode not activated")
        settings['verbose'] = False

    return settings

## Verify settings 

def _verify_settings_block(settings, block_name, block_elements):

    ## print message that we are checking settings for block_name
    pf("_verify_settings_block", "Checking settings for {0}".format(block_name))

    if block_name not in settings:
        raise ValueError("No {0} settings defined in settings file".format(block_name))
    else:
        ## Check if the nturns, npart_per_sim or beam is in simulation settings
        for key in block_elements:
            if key not in settings[block_name]:
                raise ValueError("No {0} defined in {1} settings".format(key, block_name))


def verify_settings(settings):
    """Verifies the settings in the settings file."""

    ## Print message explaining that we want to verify the settings
    pf("verify_settings", "Verifying settings")

    ## Check if LIMITLOW and LIMITHIGH are defined
    if 'LIMITLOW' not in settings or 'LIMITHIGH' not in settings:
        raise ValueError("No LIMITLOW or LIMITHIGH defined in settings file")

    _verify_settings_block(settings,                    "simulation",       ['turn_start','nturns', 'npart_per_sim', 'beam'])
    _verify_settings_block(settings,                    "beam",             ['p0c','emittance', 'mass0'])
    _verify_settings_block(settings,                    "sequence",         ['filename', 'start'])
    _verify_settings_block(settings,                    "survival",         ['activate', 'periodicity', 'filename'])
    _verify_settings_block(settings,                    "initial",          ['filename', 'path', 'read_json', 'read_dist', 'skiprows', 'skipcols', 'columns', 'add_closed_orbit'])
    _verify_settings_block(settings,                    "pulsing_pattern",  ['read', 'generate', 'apply', 'reference_values'])
    _verify_settings_block(settings["pulsing_pattern"], "read",             ['activate', 'filename'])
    _verify_settings_block(settings["pulsing_pattern"], "generate",         ['activate', 'type', 'turns_on', 'turns_off', 'seed', 'start', 'pause_length', 'pause_period'])
    _verify_settings_block(settings["pulsing_pattern"], "apply",            ['elements', 'attributes'])
    _verify_settings_block(settings,                    "dump",             ['final_dist', 'turn_by_turn', 'emittance'])
    _verify_settings_block(settings["dump"],            "turn_by_turn",     ['active', 'periodicity', 'columns', 'to_ascii'])
    _verify_settings_block(settings["dump"],            "emittance",        ['active', 'betax', 'betay', 'iterations_x', 'iterations_y', 'confidence_min', 'confidence_max', 'rolling', 'filename'])


    # check settings for HEL 
    if "HEL" in settings:
        _verify_settings_block(settings,                "HEL",  ['name', 'as_lens', 'r1', 'r2', 'length', 'current', 'voltage', 'residual_kick_x', 'residual_kick_y', 'as_aperture', 'multipole'])
        _verify_settings_block(settings["HEL"],   "multipole",  ['activate', 'knl','ksl'])
    else: 
        warnings.warn("No HEL settings defined in settings file - no HEL will be used")

    ## Check if the set_elements is in settings
    if 'set_elements' not in settings:
        warnings.warn("No set_elements defined in settings file - check if this is correct")

    ## Check if the scan is in settings
    if 'scan' not in settings:
        warnings.warn("No scan defined in settings file - check if this is correct")

    ## Check if fileList and copy_back_all are defined in settings
    if 'fileList' not in settings or 'copy_back_all' not in settings:
        raise ValueError("No fileList or copy_back_all defined in settings file")

    ## Check if the context is defined
    if 'context' not in settings:
        raise ValueError("No context defined in settings file")

    ## Check if collimation block is defined
    if 'collimation' not in settings:
        raise ValueError("No collimation defined in settings file")

    ## Check if xtrackscript is defined
    if 'xtrackscript' not in settings:
        raise ValueError("No xtrackscript defined in settings file")

    ## Check if xtracktemplates is defined
    if 'xtracktemplates' not in settings:
        raise ValueError("No xtracktemplates defined in settings file")

    ## Check if pyexe is defined
    if 'pyexe' not in settings:
        raise ValueError("No pyexe defined in settings file")

    ## Check if pyscripts is defined
    if 'pyscripts' not in settings:
        raise ValueError("No pyscripts defined in settings file")

    ## check if madx is defined
    if 'madx' not in settings:
        raise ValueError("No madx defined in settings file")




def setdefault(element, key, value):
    """Apply default values for some variables"""
    try:
        element[key]
    except KeyError:
        pf("setdefault", f"variable undefined: {key}")
        pf("setdefault", f"Setting to reference value {value}")
        element.setdefault(key, value)



def save_settings(settings, filename="settings_final.json"):

    pf("save_settings", "Saving settings in {0}".format(filename))

    settings_slim = {}

    for key in settings:
        try:
            jj = json.dumps(settings[key])
        except:
            continue
        settings_slim[key] = settings[key]
        
    jj  = json.dumps(settings_slim)
    f   = open(filename,"w")
    f.write(jj)
    f.close()

    pf("save_settings", "Settings saved in {0}".format(filename))


def pf(function_name, message, settings={"verbose": True}):
    """Print function - prints function name + message"""

    if settings['verbose']:
        print("{0:35s} - {1}".format(function_name, message))


def get_to_np(settings):
    # define function to convert to numpy in all contexts
    to_np    = settings['context_xo'].nparray_from_context_array
    pf("get_to_np", "Assigned to_np function for context {0}".format(settings['context']), settings)
    return to_np

# check which context was selected in the settings file 
def get_context(settings):
    """ Reads the context selected in the settings file
        Assigns the xo.context object to the settings 
        dictionary.

        Parameters
        ----------
        settings : dict, mandatory
            Dictionary with the simulation settings.
    """

    if settings['context'] == 'CPU':
        context = xo.ContextCpu()         # For CPU
        pf("get_context", "Context selected: CPU")

    if settings['context'] == 'GPU':
        context = xo.ContextCupy()      # For CUDA GPUs|
        pf("get_context", "Context selected: GPU (CUDA)")
    # context = xo.ContextPyopencl()  # For OpenCL GPUs

    settings['context_xo'] = context

    # return context 

########################################################################
## Tools to operate on distributions 
########################################################################

def _iteration_generate_initial_dist_particles_lost(settings, sequence, pattern, tracker=None, seed=None, turn_min=1, turn_max=201):
    
    settings['verbose'] = False
    
    local_settings = settings.copy()
    
    # seed to be used for the initial dist generation
    if seed is not None: 
        local_settings['index'] = seed
    
    # sample initial particle distribution
    part_raw = read_initial_distribution(local_settings)
    
    if not settings['initial']['sample_dist_lost']['activate']:
        return part_raw, part_raw

    # find the names of the elements in the sequence
    names = pd.Series(sequence.element_names)
    
    # get the index of the element for which we want to sample lost particles
    idx   = names[names == settings['initial']['sample_dist_lost']['element']].index[0]
    
    if tracker is None:
        # initialize tracker
        tracker = xt.Tracker(_context = settings['context_xo'], line = sequence)

    # initialize the set of particles initially used
    part_initial = part_raw.copy()   
    
    # perform the turn-by-turn tracking 
    for turn in range(turn_min, turn_max):

        _turn_idx = turn-1

        # if turn % 10 == 0:
        #     print("Tracking turn: {0} --- Particles: {1}".format(turn, len(part_raw.x[part_raw.state==1])))

        # apply the pulsing pattern 
        apply_pulsing(tracker.line, settings, pattern, turn)

        tracker.track(part_raw, num_turns=1)
        
    part_raw = part_raw.filter(part_raw.state==0)

    # select only those lost at the element of choice
    part_raw = part_raw.filter(part_raw.at_element == idx)
        
    # select only the particle IDs that were lost 
    selected_pinitial = part_initial.filter(np.isin(part_initial.particle_id, part_raw.particle_id))
        
    return selected_pinitial, part_raw



def sample_regular_grid(xmin, xmax, ymin, ymax, nx, ny):

    """Samples a regular x/y grid for FMA studies"""

    xvals  = np.linspace(xmin, xmax, nx)
    yvals  = np.linspace(ymin, ymax, ny)

    xv, yv = np.meshgrid(xvals, yvals)
    xx, yy = [], []

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            xx.append(xv[i,j])
            yy.append(yv[i,j])
            
    dfout = pd.DataFrame(xx, columns=['x'])
    dfout = dfout.assign(y = yy)

    dfout.to_string("xygrid.dat", index=False, float_format="{:,.14e}".format)



def generate_initial_dist_particles_lost(settings, sequence, pattern, turn_min=None, turn_max=None):

    if settings['initial']['sample_dist_lost']['generate']:

        # run the code to sample the distribution until the required number of particles is ready 

        pini, pfin = {}, {}

        seed      = 1

        if turn_min is None:
            turn_min = settings['simulation']['turn_start']

        if turn_max is None:
            turn_max = turn_min + settings['simulation']['nturns']

        tracker = xt.Tracker(_context = settings['context_xo'], line = sequence)
        
        while True:

            pini[seed], pfin[seed] = _iteration_generate_initial_dist_particles_lost(settings, sequence, pattern, tracker = tracker, seed=seed, turn_min=turn_min, turn_max=turn_max)

            seed += 1 

            total_particles_sampled = len(xp.Particles.merge([pini[k] for k in pini.keys()]).x)

            if total_particles_sampled > settings['simulation']['npart_per_sim']:
                break
            else:
                pf("generate_initial_dist_particles_lost", "Tracking iteration finished after {0} turns, so far sampled {1}/{2} particles".format(turn_max-turn_min, total_particles_sampled, settings['simulation']['npart_per_sim']))

        pinitial = xp.Particles.merge([pini[k] for k in pini.keys()])
        pfinal   = xp.Particles.merge([pfin[k] for k in pfin.keys()])
        
        # save the distribution 

        settings['initial']['filename'] = 'initial_dist_lost_at_{0}.json'.format(settings['initial']['sample_dist_lost']['element'].replace(".","_"))

        # remove closed orbit before saving
        if settings['initial']['add_closed_orbit']:
            pinitial = remove_closed_orbit(pinitial, settings)

        with open(settings['initial']['filename'], 'w') as fid:
            json.dump(pinitial.to_dict(), fid, cls=xo.JEncoder)


        if settings['initial']['sample_dist_lost']['stop_afterwards']:
            print("Stopping here ")
            sys.exit()


        return pinitial, pfinal
    else:
        return None, None


def read_initial_distribution(settings):
    
    """Reads initial distribution
       User has to specify the following in the settings:

       initial: 
        filename:          gpdist.selected.lost.agg
        #   filename:          final_dist.json 
        read_json:         False
        
        read_dist:         True
        skiprows:          1 
        
        # gpdist style
        columns:           ['particle_id', 'parent_particle_id', 'weight', 'x', 'y',' zeta', 'px', 'py', 'psigma',' A', 'Z', 'M', 'E', 'DE']

        if read_json is set to true, a json is read. column names must match with xpart attributes to be read
        skiprows: should be set to 1 if there is a header and 0 if no header 

    """

    # check if index is specified, otherwise use 1 
    try:
        pf("read_initial_distribution", "Index selected by user: {0}".format(settings['index']), settings)
    except KeyError:
        settings['index'] = 1 
        pf("read_initial_distribution", "Index not specified in settings file", settings)



    ## CREATE GPDIST FILE 
    if settings['initial']['create_gpdist']:
        pf("read_initial_distribution", "User selected to generate gpdist file and input on the node", settings)

        ## if user selected to generate gpdist input create it 
        generate_gpdist_input(settings)
        run_gpdist(settings)

        dist = pd.read_csv(settings['initial']['filename'],
                        delim_whitespace=True,
                        names = ['particle_id', 'parent_particle_id', 'weight', 'x', 'y',' zeta', 'px', 'py', 'psigma',' A', 'Z', 'M', 'E', 'DE']
                        )


    ## READ FROM JSON FILE 
    elif settings['initial']['read_json']:
        pf("read_initial_distribution", "User selected to read from json")
        pf("read_initial_distribution", "Filename: {0}".format(settings['initial']['filename']))
        
        # Load particles from json file to selected context
        with open(settings['initial']['filename'], 'r') as fid:
            part= xp.Particles.from_dict(json.load(fid), _context=settings['context_xo'])
    
    elif settings['initial']['read_dist']:
        pf("read_initial_distribution", "User selected to read from asci file", settings)
        # do the slicing     
        slice_distribution(
                    input_name = settings['initial']['path']+"/"+settings['initial']['filename'], 
                    islice     = settings['index'], 
                    size       = settings['simulation']['npart_per_sim'],
                    tracker    = settings['tracker'],
                    suffix     = ".slice", 
                    columns    = settings['initial']['columns'], 
                    skiprows   = settings['initial']['skiprows']
                    )
    
        input_filename = settings['initial']['filename'] + ".slice"

        dist = pd.read_csv(input_filename,  delim_whitespace=True)
       
    if not settings['initial']['read_json']:

        # now transform to xpart object 
        particle_properties = {}
        particle_properties['p0c'] = settings['beam']['p0c']
        
        # pf(function_name, message, only_verbose=False)

        for key in xp.Particles().to_dict().keys():
            if key in dist.columns:
#                 print("adding key", key)
                if key in settings['initial']['skipcols']:

                    pf("read_initial_distribution", "Key {0} in settings -> initial -> skipcols, Skipping!".format(key), 
                        settings=settings)

                    continue

                pf("read_initial_distribution", "Assigning key {0}".format(key), 
                    settings=settings)

                particle_properties[key] = dist[key]
                
        # intialize xpart object
        part = xp.Particles( _context = settings['context_xo'], **particle_properties)

        pf("read_initial_distribution", "Xpart object created", settings=settings)




    try: 
        pf("read_initial_distribution", "Assigning mass {0}".format(settings['beam']['mass0']), settings=settings)
        part.mass0 = settings['beam']['mass0']
    except:
        pf("read_initial_distribution", "User did not specify particle mass, using {0}".format(part.mass0), settings=settings)

    
    add_closed_orbit(part, settings)

    # save the initial distribution 
    pf("read_initial_distribution", "Saving distribution to dist0.json",settings=settings)
    with open('dist0.json', 'w') as fid:
        json.dump(part.to_dict(), fid, cls=xo.JEncoder)

    # "dist0.json"

    return part

########################################################################
## Tools to operate on sequences
########################################################################

def get_index_s(sequence, name):
    """Returns index and s of an element in the sequence"""
    idx = sequence.element_names.index(name)
    s   = sequence.get_s_elements()[idx]
    return idx, s

class sequence:
    def __init__(self):
        pass

    @staticmethod
    def load_sequence(settings):
        with open(settings['sequence']['filename']) as json_file:
            sequence = xt.Line.from_dict(json.load(json_file))
        pf("load_sequence", "Sequence loaded:    {0}".format(settings['sequence']['filename']))
        pf("load_sequence", "Sequence length:    {0} m".format(sequence.get_length()))
        pf("load_sequence", "Number of elements: {0}".format(len(sequence)))
        pf("load_sequence", "Old start element:  {0}".format(sequence.element_names[0]))
        sequence = sequence.cycle(name_first_element = settings['sequence']['start'])
        pf("load_sequence", "New start element:  {0}".format(sequence.element_names[0]))
        return sequence

    @staticmethod
    def install_HEL(sequence, settings):

        HEL                   = settings['HEL']

        # find HEL index in the sequence 
        idx_hel = sequence.element_names.index(HEL['name'])
        pf("install_HEL_mult", "Found HEL & index: {0} / {1}".format(HEL['name'], idx_hel))

        # names for multipole and aperture restriction at HEL
        HEL['name_elens']     = HEL['name'] + ".elens"
        HEL["name_multipole"] = HEL['name'] + ".multipole"
        HEL["name_aperture"]  = HEL['name'] + ".as_aperture"


        # Add marker for multpole component
        sequence = sequence.insert_element( index           = idx_hel + 1, 
                                            element         = xt.Elens( inner_radius    =   1,
                                                                        outer_radius    =   2,
                                                                        elens_length    =   3,
                                                                        current         =   0,
                                                                        voltage         =   1, 
                                                                        residual_kick_x =   0,
                                                                        residual_kick_y =   0
                                                                       ),
                                            name            =   HEL["name_elens"]
                                          )

        # Add marker for multpole component
        if HEL['multipole']['activate']:

            _knl = np.array(HEL['multipole']['knl'], dtype='object').astype('float')
            _ksl = np.array(HEL['multipole']['ksl'], dtype='object').astype('float')

            sequence = sequence.insert_element(index     = idx_hel + 2, 
                                            element = xt.Multipole(knl = _knl, ksl = _ksl),
                                            name    = HEL["name_multipole"])
        else:
            sequence = sequence.insert_element(index     = idx_hel + 2, 
                                            element = xt.Multipole(knl=[0], ksl=[0]),
                                            name    = HEL["name_multipole"])

        pf("install_HEL", "Adding aperture marker at idx {0}".format(idx_hel + 3))
        # Add marker for aperture check
        sequence = sequence.insert_element(index     = idx_hel + 3, 
                                        element = xt.LimitEllipse(
                                                        a = 1,
                                                        b = 1), 
                                        name    = HEL["name_aperture"])

        # if we want to simulate the E-lens as a collimator (define the halo)
        if HEL['as_aperture']:
            sequence.element_dict[HEL['name_aperture']] = xt.LimitEllipse(a = HEL['r1'], b = HEL['r1'])
            pf("install_HEL", "Assigned aperture at {0}".format(HEL['name_aperture']))
            pf("install_HEL", sequence.element_dict[HEL['name_aperture']])

        # If we want so simulate the depletion:
        if HEL['as_lens']:
            pf("install_HEL", "Setting HEL as_lens properties")
            # # introduce the hollow electron lens into the sequence
            sequence.element_dict[HEL['name_elens']] = xt.Elens(
                                                                inner_radius    =   HEL['r1'],
                                                                outer_radius    =   HEL['r2'],
                                                                elens_length    =   HEL['length'],
                                                                current         =   HEL['current'],
                                                                voltage         =   HEL['voltage'], 
                                                                residual_kick_x =   HEL['residual_kick_x'],
                                                                residual_kick_y =   HEL['residual_kick_y'])

        # if HEL['multipole']['activate']:
        #     sequence.element_dict[HEL['name_multipole']] = xt.Multipole(
        #                                                                 knl = HEL['multipole']['knl'], 
        #                                                                 ksl = HEL['multipole']['ksl']
        #                                                                 )

        pf("install_HEL", "Sequence length:    {0} m".format(sequence.get_length()))


def generate_xt_twiss(sequence, settings):
    """Generate an Xtrack twiss object from the sequence"""

    # switch off printing
    # blockPrint()

    ref_particle = xp.Particles( _context  = settings['context_xo'], p0c = settings['beam']['p0c'])
    tracker      = xt.Tracker(        line = sequence)

    sequence.particle_ref = ref_particle

    tw           = tracker.twiss()
    # enablePrint()
    
    return tw

def get_twiss(settings, sequence):
    """Calculate the twiss parameters"""
    
    # load the context 
    context = settings['context_xo']
    print("get_twiss context", context)
    
    pf("get_twiss", "Generating twiss parameters")
    # pf("get_twiss", "Context:  {}".format(settings['context_xo']))

    # define reference particle 
    ref_particle = xp.Particles( _context  = context, p0c = settings['beam']['p0c'])

    pf("get_twiss", "ref particle context {0}".format(ref_particle._buffer.context))

    ref_particle.reorganize()

    # blockPrint()
    tracker      = xt.Tracker(  _context=context,      line = sequence)
    # enablePrint()

    # specify the reference particle
    sequence.particle_ref = ref_particle
    sequence.particle_ref.reorganize()

    # Run the twiss command 
    tw           = tracker.twiss()

    # get the closed orbit 
    get_closed_orbit_start(settings, tracker, ref_particle)

    # create dataframe with the twiss parameters  vs. s
    keys = ['name', 's', 'betx', 'bety', 'alfx', 'alfy', 'x', 'px', 'y', 'py', 'gamx', 'gamy', 'dx', 'dpx', 'dy', 'dpy', 'mux', 'muy']

    # twiss_parameters = { your_key: to_np(tw[your_key]) for your_key in keys }

    twiss_parameters = {}
    for key in keys:
        # print(key)
        # print(tw[key])
        twiss_parameters[key] = tw[key]

    # print(twiss_parameters)

    twiss_parameters = pd.DataFrame(twiss_parameters)
    
    if settings['sequence']['save_twiss']:
        pf("get_twiss", "User requested saving of Twiss parameters to optics.tfs")  
        twiss_parameters.to_string("optics.tfs", index=False)

    # pf("get_twiss", "Saving tunes to tunes.json")
    tunes_keys = ['qx', 'qy', 'qs', 'dqx', 'dqy']
    tunes = { your_key: tw[your_key] for your_key in tunes_keys }
    # tunes['gamma'] = to_np(ref_particle.gamma0)[0]


    # print tune and chroma info for the user 
    pf("get_twiss", "Tunes and chromaticities")

    for kt in tunes.keys():
        pf("get_twiss", "{0}: {1}".format(kt, tunes[kt]))
    
    # jj  = json.dumps(tunes)
    # f   = open("tunes.json","w")
    # f.write(jj)
    # f.close()
    
    # for key in tunes:
    #     settings['beam'][key] = tunes[key]
        
    gamma0 = settings['closed_orbit']['gamma0'][0]

    pf("get_twiss", "Sourcing gamma0 from settings: {0}".format(gamma0))

    # calculate normalized emittance 
    pf("get_twiss", "Calculating geometric emittance")
    settings['beam']['ex'] = settings['beam']['emittance'][0]/gamma0
    settings['beam']['ey'] = settings['beam']['emittance'][1]/gamma0
    
    pf("get_twiss", "ex: {0}".format(settings['beam']['ex']))
    pf("get_twiss", "ey: {0}".format(settings['beam']['ey']))

    settings['twiss'] = twiss_parameters

    # pf("get_twiss", "Done")

    return twiss_parameters


def normalize_coordinates(coord6d, twiss):
    """Function to normalize the coordinates in 6d space"""

    # r matrix in physical space 
    Rphy = twiss['R_matrix']
    
    lnf  = xt.linear_normal_form
    
    W, invW, R = lnf.compute_linear_normal_form(Rphy)
    
    # get closed orbit 
    clo = [getattr(twiss['particle_on_co'], a)[0] for a in ['x','px','y','py','zeta','delta']]
    clo = np.array(clo)
    
    coord6d = coord6d - clo
    
    return np.dot(invW, coord6d)




class collimation:
    def __init__(self, settings, twiss):
        self.filename    = settings['collimation']['colldb']
        self.settings    = settings
        pf("collimation init", "CollDB: {0}".format(self.filename))
        
        self.twiss       = twiss
        self.collimators = []
        self.families    = {}
        
    def assign_families(self):
        additional_settings = False

        with open(self.filename) as f:
            for line in f:
                if line.startswith("#"):
                    continue

                if line.startswith("NSIG_FAM"):
#                     print("Assigning properties for collimator family {0}".format(line.split()[1]))
                    self.families[line.split()[1]] = {}
                    self.families[line.split()[1]]['sigma'] = float(line.split()[2])
                    self.families[line.split()[1]]['type']  = line.split()[3]
                    continue 

                if line.startswith("SETTINGS"):
#                     print("Settings activated")
                    additional_settings = True 
                    continue 

                if not additional_settings:           
                    _name    = line.split()[0]
                    try:
                        _setting_sig = float(line.split()[1])
                    except ValueError:
                        _setting_sig = self.families[line.split()[1]]['sigma'] 

                    _material  = line.split()[2]               
                    _length    = float(line.split()[3])
                    _angle     = float(line.split()[4])
                    _offset    = float(line.split()[5])

                    self.collimators.append([_name, _setting_sig, _material, _length, _angle, _offset])

        self.collimators = pd.DataFrame(self.collimators, columns=['name','opening_sigma', 'material','length','angle','offset'])
        
    def get_geometric_halfgaps(self):

        to_np        = get_to_np(self.settings)

        tw_coll = self.twiss[['name','s','betx','bety','x','y']]
        
        self.collimators = self.collimators.merge(tw_coll, on='name')

        class SkewNotImplemented(Exception):
            pass

        # get the number of skew collimators
        skews = self.collimators.query('((angle>0) and (angle <90)) or ((angle >90) and (angle <180))')

        if len(skews) > 0:
            raise SkewNotImplemented('Found Skew collimators. Not yet implemented!\n {0}'.format(skews[['name', 'angle']]))
        else:
            pf("get_geometric_halfgaps", "No Skew collimators found, continuing", self.settings)
            # print(skews)
            
        sigx = np.sqrt(self.settings['beam']['ex']*self.collimators['betx'])
        sigy = np.sqrt(self.settings['beam']['ey']*self.collimators['bety'])
        
        self.collimators = self.collimators.assign(sigx = sigx)
        self.collimators = self.collimators.assign(sigy = sigy)

        angle_rad = self.collimators['angle']*(2*np.pi/360)
        
        hgap_x    = (1/np.cos(angle_rad))*sigx*self.collimators['opening_sigma']
        hgap_y    = (1/np.sin(angle_rad))*sigy*self.collimators['opening_sigma']
        
        self.collimators = self.collimators.assign(hgap_x = hgap_x)
        self.collimators = self.collimators.assign(hgap_y = hgap_y)
        
        self.collimators = self.collimators.replace([np.inf, -np.inf], 1e16)
        
    def get_jaw_positions(self):
        
        positive_jaw_x = self.collimators.x + self.collimators.hgap_x 
        negative_jaw_x = self.collimators.x - self.collimators.hgap_x 
        
        positive_jaw_y = self.collimators.y + self.collimators.hgap_y 
        negative_jaw_y = self.collimators.y - self.collimators.hgap_y 
        
        self.collimators = self.collimators.assign(positive_jaw_x = positive_jaw_x)
        self.collimators = self.collimators.assign(negative_jaw_x = negative_jaw_x)
        
        self.collimators = self.collimators.assign(positive_jaw_y = positive_jaw_y)
        self.collimators = self.collimators.assign(negative_jaw_y = negative_jaw_y)
        
        pf("get_jaw_positions", "Saving collimator settings to coll_settings.dat")
        self.collimators.to_string("coll_settings.dat")
        
        self.collimators.index = self.collimators.name
        pass
    
    def add_collimator_settings(self, halfgaps=None):
        
        self.assign_families()

        # # halfgaps should be dict
        if halfgaps is not None:
            for cname in halfgaps.keys():
                # apply the new value of the collimator setting
                self.collimators['opening_sigma'].mask(self.collimators['name'] == cname, halfgaps[cname], inplace=True)

        self.get_geometric_halfgaps()
        self.get_jaw_positions()

        cc_dict = self.collimators.to_dict()
        self.settings['coll_settings'] = cc_dict
#         self.settings[]
        return self.settings 


    @staticmethod
    def install_all_collimator_markers(sequence, settings):
        for name in settings['coll_settings']['name'].keys():
            collimation.install_collimator_markers(sequence, settings, name)

    @staticmethod
    def install_collimator_markers(sequence, settings, name):
        """Takes the sequence and adds collimator markers upstream and downstream, depending on the length of the element"""

        clength = settings['coll_settings']['length'][name]
        _, s  = get_index_s(sequence, name)

        pf("install_collimator_markers", "Installing collimator {0} of length {1} at s={2}".format(name, clength, s), settings)

        
        # install upstream marker if it didn't exist yet
        if name + "..U" not in sequence.element_names:
            sequence.insert_element(element = xt.Drift(), name = name + "..U", at_s = s - clength/2)
            pf("install_collimator_markers", "Upstream marker {0} at {1} installed".format(name + "..U", s - clength/2), settings)
        else:
            pf("install_collimator_markers", "Upstream marker {0} at {1} exists".format(name + "..U", s - clength/2), settings)
            pf("install_collimator_markers", sequence.element_dict[name + "..U"].to_dict(), settings)


        # install downstream marker if it didn't exist
        # index has changed - need to re-load
        _, s  = get_index_s(sequence, name)

        if name + "..D" not in sequence.element_names:
            sequence.insert_element(element = xt.Drift(), name = name + "..D", at_s = s + clength/2)
            pf("install_collimator_markers", "Downstream marker {0} at {1} installed".format(name + "..D", s + clength/2), settings)
        else:
            pf("install_collimator_markers", "Downstream marker {0} at {1} exists".format(name + "..D", s + clength/2), settings)
            pf("install_collimator_markers", sequence.element_dict[name + "..D"].to_dict(), settings)

        return sequence
    
    @staticmethod
    def apply_collimator_settings(sequence, settings):
        ## apply collimator settings 

        # def apply_collimator_settings

        for cname in settings['coll_settings']['name'].keys():
            for ename in sequence.element_names:
                if cname in ename:

                    if "..U" not in ename and "..D" not in ename:
                        continue

                    _limits = {}

                    _limits['min_x'] = settings['coll_settings']['negative_jaw_x'][cname]
                    _limits['max_x'] = settings['coll_settings']['positive_jaw_x'][cname]

                    _limits['min_y'] = settings['coll_settings']['negative_jaw_y'][cname]
                    _limits['max_y'] = settings['coll_settings']['positive_jaw_y'][cname]

                    for key in _limits.keys():
                        if abs(_limits[key]) < 0.1:

                            if "_x" in key:
                                suffix = 'x'
                            else:
                                suffix = 'y' 

                            _beta = settings['coll_settings']['bet'+suffix][cname]
                            _sig  = settings['coll_settings']['sig'+suffix][cname]
                            _nsig = settings['coll_settings']['opening_sigma'][cname]

                            if ("..U" in ename) and ('max' in key):
                                pf("apply_collimator_settings", "Setting {0:20s} {1} to {2: .2f} mm / bet{3} = {4:4.0f}m / sig{3} = {5}um / nsig = {6}".format(ename.split("..")[0], key, 
                                    round(_limits[key]*1e3,2), suffix,  round(_beta,0), round(_sig*1e6), _nsig
                                    ))
                    
                    # print()
                    sequence.element_dict[ename] = xt.LimitRect(**_limits)
                    # print(" ")
                    
        return sequence 

    @staticmethod

    # aggregate the particle losses at the collimators 
    def write_coll_summary(particles, sequence, settings):
        """Creates the coll_summary table and writes to a file """

        to_np        = get_to_np(settings)

        _ps          = to_np(particles.state).copy()
        _at_element  = to_np(particles.at_element).copy()
        _at_turn     = to_np(particles.at_turn).copy()
        _pid         = to_np(particles.particle_id).copy()

        # find the particles that were lost
        idx_lost     = (_ps == 0)

        # get the indices of the elements at which the particles are lost
        elidx_lost   = _at_element[idx_lost]

        # _elnames     = to_np(sequence.element_names).copy()
        elnames      = np.array(sequence.element_names)

        # get the collimator at which the particles are lost
        coll_lost  = elnames[elidx_lost]

        # rename the collimators -> remove the ..
        coll_lost  = [n.split("..")[0] for n in coll_lost]

        # get the turn at which the individual particles were lost
        turn_lost  = _at_turn[idx_lost]

        # get the ID of the particles lost
        pid_lost  = _pid[idx_lost]

        # detailed dataframe
        collsum_detailed = pd.DataFrame([])
        collsum_detailed = collsum_detailed.assign(name            = coll_lost)
        collsum_detailed = collsum_detailed.assign(nimp            = 1 )
        collsum_detailed = collsum_detailed.assign(turn            = turn_lost)
        collsum_detailed = collsum_detailed.assign(pid             = pid_lost)

        # coll_summary
        collsum = collsum_detailed.groupby('name').sum()
        collsum = collsum.reset_index()
        collsum = collsum[['name','nimp']]

        # write output files
        collsum_detailed.to_string('coll_losses.dat', index=False)

        collsum.to_string('coll_summary.dat', index=False)


class cavities:
    def __init__(self):
        pass
    
    @staticmethod
    def apply_settings(sequence, settings):
        for cav in sequence.get_elements_of_type(xt.Cavity)[1]:

            pf("cavities-apply_settings", "Applying cavity settings to {0}".format(cav), settings)

            idx = sequence.element_names.index(cav)   
            sequence.elements[idx].voltage   = settings['cavities']['voltage']
            sequence.elements[idx].frequency = settings['cavities']['frequency']
            sequence.elements[idx].lag       = settings['cavities']['lag']



def get_element_type_string(xt_element):
    tt  = type(xt_element)
    tt  = str(tt).split(".")[-1].replace(">","").replace("'","")
    return tt

def save_tracked_sequence(sequence):
    """Save a sequence in .dat and .json file"""

    pf("save_tracked_sequence", "Saving sequence to {0}".format('tracked_sequence.dat')) 

    seq_out = pd.DataFrame([])
    seq_out = seq_out.assign(name = sequence.element_names)
    seq_out = seq_out.assign(s    = sequence.get_s_elements())
    seq_out = seq_out.assign(element    = sequence.elements)
    seq_out = seq_out.assign(element_type = seq_out.apply(lambda x : get_element_type_string(x['element']), axis=1))
    seq_out = seq_out.drop(columns=['element'], axis=1)
    seq_out.to_string('tracked_sequence.dat')

    pf("save_tracked_sequence", "Saving sequence to {0}".format('tracked_sequence.json')) 

    # Save to json
    with open('tracked_sequence.json', 'w') as fid:
        json.dump(sequence.to_dict(), fid, cls=xo.JEncoder)

    



## SURVIVAL

class survival:

    def __init__(self, settings):


        self.qsurvival   = settings['survival']['activate']

        if self.qsurvival:
            pf("class: survival", "User switched on generation for survival.dat")
        else:
            pf("class: survival", "User switched off generation for survival.dat")
            return 

        self.survival    = []
        self.filename    = settings['survival']['filename']
        self.periodicity = settings['survival']['periodicity']
        self.extra_data  = None
        self.to_np       = get_to_np(settings)

        pass
    
    def get_number_of_particles(self, particles):
        """Calculates the number of particles still in the tracking"""
        ps           = self.to_np(particles.state).copy()   # PARTICLE STATE
        npart_surv   = ps[ps>=0].sum()                 # take the sum 
        return npart_surv    

    def get_surviving_particles(self, tt, particles, extra_data=None):
        """Append to the array"""
        if not self.qsurvival:
            return

        if (tt % self.periodicity == 0) or (tt==1):

            # Get the number of particles that are still in the tracking
            surviving_turn_i = self.get_number_of_particles(particles)

            if extra_data is None:
                self.survival.append([tt, surviving_turn_i])

            else:

                to_append = [tt, surviving_turn_i]

                for k in extra_data.keys():
                    to_append.append(extra_data[k])

                self.extra_data = extra_data

                self.survival.append(to_append)

            # pf("class: survival", "\rTracking turn {0}: particles {1}".format(tt, surviving_turn_i))
            
    def write_survival(self):
        """Write survival.dat file """

        if not self.qsurvival:
            return

        pf("survival", "Writing to {0}".format(self.filename))

        if self.extra_data is None:
            survpd = pd.DataFrame(self.survival, columns = ['turn','surv'])

        else:
            columns = ['turn', 'surv']
            for key in self.extra_data.keys():
                columns.append(key)

            pf("survival", "User requested to write additional data columns")
            pf("survival", "All columns to write are: {0}".format(columns))
            survpd = pd.DataFrame(self.survival, columns = columns)
            

        survpd.to_string(self.filename, index=False)


class emittance:
    def __init__(self, settings):
        self.settings    = settings['dump']['emittance']
        self.periodicity = 1
        self.x           = [] 
        self.y           = []
        self.ex, self.ey = [], []
        self.to_np       = get_to_np(settings)

        if not settings['dump']['emittance']['active']:
            pf("class: emittance", "User did not select to calculate emittance")
            self.active = False
            return            
        else:
            self.active = True
        
    def add_samples(self, p):
               
        xi     = self.to_np(p.x).copy()
        yi     = self.to_np(p.y).copy()
        statei = self.to_np(p.state).copy()
        
        # select only those particles with state 1
        xi = xi[statei ==1]
        yi = yi[statei ==1]
        
        # attach particles to the array 
        self.x.append(xi)
        self.y.append(yi)
        
        self.gamma = p.gamma0[0]
        
    def reset_samples(self):
        self.x = []
        self.y = []
        
    def _get_emittance(self, turn, plane):
               
        gamma = self.gamma
        
        confidence_min = self.settings['confidence_min']
        confidence_max = self.settings['confidence_max']
        
        # get the number of iterations for the bootstrapping
        if plane == 'horizontal':
            n_iterations   = self.settings['iterations_x']
            amplitude      = np.concatenate(self.x)            
            beta           = self.settings['betax']
            
        elif plane == 'vertical':
            n_iterations   = self.settings['iterations_y']
            amplitude      = np.concatenate(self.y)
            beta           = self.settings['betay']
        
        # get the initial rms
        sigma_rms  = get_rms(amplitude)

        # do the bootstrapping
        yboot      = bootstrap(amplitude, get_rms, n_iterations = n_iterations)

        # get the confidence levels
        qmin       = np.percentile(yboot, confidence_min)
        qmax       = np.percentile(yboot, confidence_max)

        # get the rms of the observed data points
        eyn      = 1e6*gamma*sigma_rms**2/(beta)

        # get the rms of the confidence levels
        ey_qmin  = 1e6*gamma*qmin**2/(beta)
        ey_qmax  = 1e6*gamma*qmax**2/(beta)

        
        if plane == 'horizontal':
            self.ex.append([turn, eyn, ey_qmin, ey_qmax])

        if turn % 50 == 0 : 
            print("Vertical emittance turn {0}: {1}".format(turn, [turn, eyn, ey_qmin, ey_qmax]))

        if plane == 'vertical':
            self.ey.append([turn, eyn, ey_qmin, ey_qmax])
        
        
        
    def get_emittance(self,
                      turn,
                      particles):
        """Get the normalized emittances in murad incl. confidence level from bootstrapping."""

        if not self.active:
            return 
        
        # if the user selects to calculate the emittance at 
        # each turn:
        # settings['rolling'] = 1

        if self.settings['rolling'] == 1:
            self.add_samples(particles)
            self._get_emittance(turn, 'horizontal')
            self._get_emittance(turn, 'vertical')
            self.reset_samples()
            return 
        
        # if the user selects to aggregate over several turns:
        # settings['rolling'] > 1

        if turn == 0:
            self.add_samples(particles)
            return 
                   
        if turn % self.settings['rolling'] != 0 :
            self.add_samples(particles)
        else:
            self._get_emittance(turn, 'horizontal')
            self._get_emittance(turn, 'vertical')
            self.reset_samples()
            
    
    def write_emittance_table(self):

        """Writes files with the emittance evolution in the two planes
        """

        settings = self.settings
        
        if not self.active:
            return

        # check if the user has given a filename to be used
        try:
            emit_fn = settings['filename']
        except:
            emit_fn = 'emittance.dat'

        emittances_x, emittances_y = self.ex, self.ey
            
        emittances_x = pd.DataFrame(emittances_x, columns=['turn','emit_rms', 'emit_lowerCL', 'emit_upperCL'])
        emittances_x.to_string(emit_fn.split(".")[0] + "_x." + emit_fn.split(".")[1], index=False)

        emittances_y = pd.DataFrame(emittances_y, columns=['turn','emit_rms', 'emit_lowerCL', 'emit_upperCL'])
        emittances_y.to_string(emit_fn.split(".")[0] + "_y." + emit_fn.split(".")[1], index=False)

        pf("class: emittance", "emittance tables written")




class dump:
    def __init__(self, settings):

        self.settings = settings['dump']['turn_by_turn']

        # check if the dump module is active 
        if 'active' in self.settings.keys():

            if self.settings['active']:
                self.active = True
                pf("class: dump", "Run script - DUMP module - User selected to activate turn by turn dumping")
            
            else:
                pf("class: dump", "Run script - DUMP module - User selected to deactivate turn by turn dumping")
                self.active = False
                return 

        else:
            pf("class: dump", "Run script - DUMP module - User selected to deactivate turn by turn dumping")
            self.active = False
            return 

        if 'columns' not in self.settings.keys():
            pf("class: dump", "Run script - DUMP module - User did not specify columns in settings file")
            self.columns = ['x','px','y','py','delta','zeta','particle_id']
        else:
            pf("class: dump", "Run script - DUMP module - User specified the following columns for dump: {0}".format(self.settings['columns']))
            self.columns = self.settings['columns']

        pf("class: dump", "Run script - DUMP module - initializing database")
        self.conn = sqlite3.connect('dump.sqlite')

    def check_dump(self, turn):
        """Checks if this turn should be dumped"""
        dump_this_turn = True

        # if user has given a certain periodicity to dump 
        if 'periodicity' in self.settings.keys():
            dump_this_turn = False
            periodicity = int(self.settings['periodicity'])
            if turn % periodicity == 0: 
                dump_this_turn = True

        return dump_this_turn

    def write_dumpfile(self):

        if not self.active:
            return 

        self.conn.close()

        if self.settings['to_ascii']:
            pf("write_dumpfile", "User selected to write ascii file")
            dump.save_dump_asci()


    @staticmethod
    def save_dump_asci():
        cnx = sqlite3.connect('dump.sqlite')
        df = pd.read_sql_query("SELECT * FROM dump_turn", cnx)
        df.to_string("dump.dat", index=False, float_format="{:,.14e}".format)


    def dump_turn(self, particles, turn, to_np):

        if not self.active:
            return 

        if not self.check_dump(turn):
            return

        p       = particles
        dfpart  = pd.DataFrame([])

        for attrname in self.columns:

            # get the attribute, for example x, px, etc.
            _var = getattr(p, attrname)

            _var = to_np(_var)

            # prepare the assigning of the new variable
            kwargs = {attrname : _var}

            dfpart = dfpart.assign(**kwargs)
        
        dfpart = dfpart.assign(state = to_np(p.state))
        dfpart = dfpart.assign(turn  = turn)

        dfpart = dfpart.query("state>0")

        # remove the state
        dfpart = dfpart.drop(columns=['state'], axis=1)

        # add data to the sqlite database  
        dfpart.to_sql('dump_turn', self.conn, if_exists='append', index = False)

        if turn % 100 == 0:
            pf("class: dump", "Run script - DUMP module - Turn {0} added to sqlite file".format(turn))



# functions to prepare for the pulsing -> get the default values that are going to be used 
# as a baseline for the pulsing

def add_all_reference_settings(line, settings):
    for i, element_name in enumerate(settings['pulsing_pattern']['apply']['elements']):
        for ia, attribute in enumerate(settings['pulsing_pattern']['apply']['attributes'][i]):
            _add_reference_settings(line, settings, element_name, attribute)

def _add_reference_settings(line, settings, name, attr):

    if 'reference_values' not in settings['pulsing_pattern'].keys():
        settings['pulsing_pattern']['reference_values'] = {}

    if name not in settings['pulsing_pattern']['reference_values'].keys():
        settings['pulsing_pattern']['reference_values'][name] = {}


    # use the value from the sequence
    if attr not in settings['pulsing_pattern']['reference_values'][name].keys():

        pf("add_reference_settings", "Attribute not specified in settings.yml")
        pf("add_reference_settings", "Adding reference settings for {0}: {1}".format(name, attr))

        settings['pulsing_pattern']['reference_values'][name][attr] = line.element_dict[name].to_dict()[attr]
    
    else:
        pf("add_reference_settings", "Attribute specified in settings.yml {0}/{1}/{2}".format(name, attr, settings['pulsing_pattern']['reference_values'][name][attr]))

    
        
    # settings['pulsing_pattern']['reference_values'][name][attr] = getattr(line.element_dict[name], attr)



def generate_adt_time_profile(settings, gain_ramp_time = None, rampdown_final_level = None, gain_ramp_start_level = None):
    """Generate an ADT time profile for the quench test. 
    Later rename to adt white_noiser
    gain_ramp_time        : number of seconds until the maximum should be reached
    rampdown_final_level  : level between zero and one where we want to stop the adt time profile   
    """

    nturns   = settings['simulation']['nturns']

    if gain_ramp_start_level is None:
        try:
            gain_ramp_start_level       = settings['pulsing_pattern']['ramp_adt']['gain_ramp_start_level']
            gain_ramp_start_level       = float(gain_ramp_start_level)
        except:
            # backwards compatibility
            pf('generate_adt_time_profile', 'WARNING: Did not find gain_ramp_start_level in settings file')
            pf('generate_adt_time_profile', '         Setting gain_ramp_start_level to zero')


    if gain_ramp_time is None:
        gain_ramp_time       = settings['pulsing_pattern']['ramp_adt']['gain_ramp_time']
        gain_ramp_time       = int(gain_ramp_time)

    if rampdown_final_level is None:
        rampdown_final_level = settings['pulsing_pattern']['ramp_adt']['rampdown_final_level']
        rampdown_final_level = float(rampdown_final_level)

    try: 
        if settings['pulsing_pattern']['ramp_adt']['from_file'] is False: 
            from_file = False
        else:
            from_file = settings['pulsing_pattern']['ramp_adt']['from_file']
    except:
        from_file=False

    # sample the white noise from the ADT (values between 0 and 1)
    adt_kick = (np.random.uniform(low=0.0, high=1.0, size=nturns)-0.5)/0.5

    if not from_file:
        
        turns_per_second = 11245

        turns_to_peak      = turns_per_second*gain_ramp_time

        ramp_to_peak       = np.linspace(gain_ramp_start_level,1,turns_to_peak)

        rampdown_from_peak = np.linspace(1,rampdown_final_level, max(1,nturns - turns_to_peak))

        gain_ramp = np.append(ramp_to_peak, rampdown_from_peak)

    else: 
        # if we read from file 
        gain_ramp = pd.read_csv(from_file, names=['modulation'])
        gain_ramp = np.array(gain_ramp['modulation'])

    # if we want to start after a couple of turns 
    try:
        if int(settings['pulsing_pattern']['ramp_adt']['start'])>0:
            start_zeros = np.zeros(settings['pulsing_pattern']['ramp_adt']['start'])
            gain_ramp   = np.append(start_zeros, gain_ramp)
    except KeyError:
        pass

    # to match the lengths 
    gain_ramp = gain_ramp[:len(adt_kick)]

    df = pd.DataFrame(adt_kick*gain_ramp, columns=['adt_kick'])
    df.to_string("adt_kicks_per_turn.dat")

    return adt_kick*gain_ramp


def run_gpdist(settings):

    # if not settings['initial']['gpdist']['activate']:
    #     return

    beamline_element_name = settings['sequence']['start'].replace(".","_")

    if settings['initial']['gpdist_type'] == 'dgauss':
        pf("run_gpdist", "Running gpdist for double gaussian", settings)
        subprocess.run(["./gpdist.exe",   'gpdist_dgauss_{0}.dat'.format(beamline_element_name)], capture_output=True)
        settings['initial']['filename'] = "initial.dgauss.{0}.dat".format(beamline_element_name)
    

    if settings['initial']['gpdist_type'] == 'gauss':
        pf("run_gpdist", "Running gpdist for gaussian", settings)
        subprocess.run(["./gpdist.exe", 'gpdist_gauss_{0}.dat'.format(beamline_element_name)],  capture_output=True)
        settings['initial']['filename'] = "initial.gauss.{0}.dat".format(beamline_element_name)

    pf("run_gpdist", "Setting filename for initial distribution: {0}".format(settings['initial']['filename']), settings)




def generate_gpdist_input(settings):
    # generate gpdist file to generate initial distribution 

    # if not settings['initial']['create_gpdist']:
    #     return

    tw = settings['twiss']

    # tw = twiss 
    reference_emit = float(settings['beam']['emittance'][0])

    optstart = tw.iloc[0]

    beamline_element_name = settings['sequence']['start'].replace(".","_")
    # initial.dgauss.{2}.dat

    try:
        sigma_cut = settings['initial']['sample_dist_lost']['sigma_cut']
        pf("generate_gpdist_input", "Using sigma cut: {0}".format(sigma_cut))
    except:
        sigma_cut = [0,10]

    with open('gpdist_dgauss_{0}.dat'.format(beamline_element_name), 'w') as f:
        f.write("PARAM     1   {0:.8f} {1:.8f} {2:.8f} {3:.8f} {4:.8f} \n".format(optstart['betx'], optstart['alfx'], optstart['dx'], optstart['dpx'], reference_emit))
        f.write("PARAM     2   {0:.8f} {1:.8f} {2:.8f} {3:.8f} {4:.8f} \n".format(optstart['bety'], optstart['alfy'], optstart['dy'], optstart['dpy'], reference_emit))
        f.write("PARAM     3   0.938272081   {0}    0.0         0.0\n".format(settings['beam']['p0c']/1e9))
        
        
        f.write("DTYPE     1         BIGAUS     {0}    {1}   1.0    2.0    0.65 \n".format( sigma_cut[0], sigma_cut[1]))
        f.write("DTYPE     2         BIGAUS     {0}    {1}   1.0    2.0    0.65 \n".format( sigma_cut[0], sigma_cut[1]))
        f.write("DTYPE     3         PENCIL \n")
        
        f.write("OFFSET   0.E-00   0.E-00   0.E-00   0.0E-00   0.E+00   0.0E-00 \n")
        
        f.write("OUTPUT    {0}      {1}    initial.dgauss.{2}.dat    header.dat     0    0".format(
            settings['simulation']['npart_per_sim'], 
            settings['index'], 
            beamline_element_name))
        

    with open('gpdist_gauss_{0}.dat'.format(beamline_element_name), 'w') as f:
        f.write("PARAM     1   {0:.8f} {1:.8f} {2:.8f} {3:.8f} {4:.8f} \n".format(optstart['betx'], optstart['alfx'], optstart['dx'], optstart['dpx'], reference_emit))
        f.write("PARAM     2   {0:.8f} {1:.8f} {2:.8f} {3:.8f} {4:.8f} \n".format(optstart['bety'], optstart['alfy'], optstart['dy'], optstart['dpy'], reference_emit))
        f.write("PARAM     3   0.938272081   {0}    0.0         0.0\n".format(settings['beam']['p0c']/1e9))
        
        f.write("DTYPE     1         GAUSS     {0}    {1} \n".format( sigma_cut[0], sigma_cut[1]))
        f.write("DTYPE     2         GAUSS     {0}    {1} \n".format( sigma_cut[0], sigma_cut[1]))   
        f.write("DTYPE     3         PENCIL \n")
        
        f.write("OFFSET   0.E-00   0.E-00   0.E-00   0.0E-00   0.E+00   0.0E-00 \n")
        
        f.write("OUTPUT    {0}      {1}    initial.gauss.{2}.dat    header.dat     0    0".format(
            settings['simulation']['npart_per_sim'], 
            settings['index'], 
            beamline_element_name))




def apply_pulsing(line, settings, pattern, turn, values_key='reference_values'):
    
    """
        values_key: name of the block in the settings file that should be used for the pulsing - useful in case multiple pulsing mechanisms should be used in parallel
    """
    
    # if pulsing is not activated 
    if settings['pulsing_pattern']['activate'] is False:
        return 


    pref     = settings['pulsing_pattern'][values_key]
    idx_turn = turn - 1                                            
    on_off   = pattern[idx_turn]                                   # state of the pulsing (on/off)
    
    for name in pref.keys():
        
        for attr in pref[name].keys():

            _val = pref[name][attr]

            if (type(_val) is list) or (type(_val) is np.ndarray):
                
                if settings['context']=='GPU':

                    import cupy 
                    _val = [float(v)*on_off for v in _val]
                    _val = cupy.array(_val)
                    setattr(line.element_dict[name], attr, _val)

                    pf("apply_pulsing", "Turn {0} : applying pulsing {1}, {2}, {3}".format(turn, name,  attr, [float(v)*on_off for v in _val]), settings)

                else:
                    setattr(line.element_dict[name], attr, [float(v)*on_off for v in _val])
                    
                    pf("apply_pulsing", "Turn {0} : applying pulsing {1}, {2}, {3}".format(turn, name,  attr, [float(v)*on_off for v in _val]), settings)
            else:
                setattr(line.element_dict[name], attr, _val*on_off)

                pf("apply_pulsing", "Turn {0} : applying pulsing {1}, {2}, {3}".format(turn, name,  attr, _val*on_off), settings)

            


def get_closed_orbit_start(settings, tracker, ref_particle):
    """Closed orbit information that can be used to offset the initial distribution"""   
    pf("get_closed_orbit_start", "Writing closed orbit to settings object") 

    clo = tracker.find_closed_orbit(particle_ref = ref_particle)

    settings['closed_orbit'] = clo.to_dict()

    for key in settings['closed_orbit'].keys():
        try:
            pf("get_closed_orbit_start", "Closed orbit {0:19s} : {1}".format(key, settings['closed_orbit'][key][0]))
        except IndexError:
            pf("get_closed_orbit_start", "Closed orbit {0:19s} : {1}".format(key, settings['closed_orbit'][key]))

def add_closed_orbit(particles, settings):
    """Adds the closed orbit to the particle coordinates
       Usage:
            add_closed_orbit(particles, settings)
    
    """

    if not settings['initial']['add_closed_orbit']:
        pf("add_closed_orbit", "User deactivated adding closed orbit!")

    for key in ['x','y','px','py','delta','zeta']:

        pf("add_closed_orbit", "Adding {0} closed orbit: {1}".format(key, settings['closed_orbit'][key][0]), settings=settings)
        pf("add_closed_orbit", "Initial {0}: {1}".format(key, getattr(particles, key)[0]), settings=settings)

        getattr(particles, key)[:] += settings['closed_orbit'][key][0]
  
        pf("add_closed_orbit", "Final   {0}: {1}".format(key, getattr(particles, key)[0]), settings=settings)
        


def remove_closed_orbit(particles, settings):
    """Removes the closed orbit from the particle coordinates
       Usage:
            remove_closed_orbit(particles, settings)
    
    """

    part = particles.copy()

    for key in ['x','y','px','py','delta','zeta']:
        
        pf("remove_closed_orbit", "Initial {0:6s}: {1}".format(key, getattr(particles, key)[0]), settings)

        _value_updated = getattr(part, key) - settings['closed_orbit'][key][0]
        
        pf("remove_closed_orbit", "Final   {0:6s}: {1}".format(key, _value_updated[0]), settings)
        
        setattr(part, key, _value_updated)
           
    return part


def set_element_value(sequence, settings):

    """Sets an element attribute to the reference value defined 
       in the settings file. 
    """

    if 'set_elements' not in settings.keys():
        pf("set_element_value", "User did not add 'set_elements' in settings file")
        return 


    for i, element_name in enumerate(settings['set_elements']['names']):
        key   = settings['set_elements']['keys'][i]
        value = settings['set_elements']['values'][i]
        
        if isinstance(value, list):
            value = np.array(value).astype('float')
            value = value.tolist()

        pf("set_element_value", "setting element {0} attribute {1} to value {2}".format(element_name, key, value))
        
        setattr(sequence.element_dict[element_name], key, value)


## save to gpdist object
def part_to_gpdist(part, name="xtpart.gpdist"):
    
    """Saves particle object to gpdist style"""
    
    gpdist = pd.DataFrame([])

    cols   = ['particle_id', 'parent_particle_id', 'weight', 'x', 'y','zeta', 'px', 'py', 'psigma','A', 'Z', 'M', 'E', 'DE']

    dict_to_gpdist = {}

    for k in cols:
    #     print(k)
        if k in part.to_dict().keys():
    #         print(k)
            dict_to_gpdist[k] = part.to_dict()[k]

    dict_to_gpdist["E"]      = np.sqrt(part.mass0**2 + part.p0c**2)*1e-9
    dict_to_gpdist["M"]      = part.mass0*1e-9
    dict_to_gpdist["A"]      = 1
    dict_to_gpdist["Z"]      = 1
    dict_to_gpdist["weight"] = 1
    dict_to_gpdist["DE"]     = 0

    gpdist = gpdist.assign(**dict_to_gpdist)
    gpdist = gpdist[cols]
    
    gpdist.to_string(name, index=False, header=False, float_format="{:,.14e}".format)



# options: simultaneously, sequentially, independently, all


class scraping:

    def __init__(self, settings, sequence):

        self.settings = settings 

        self.active   = self.settings['scraping']['activate']

        try:
            self.initial_filename = self.settings['scraping']['initial']
        except:
            self.initial_filename = 'final_dist.json'
                
        # settings to read the initial distribution 
        self.settings['initial'] = {"filename":  self.initial_filename, 
                                    "read_json": True, "read_dist": False, "create_gpdist": False, "add_closed_orbit": False }

        self.reset_initial()
        self.survival   = survival(self.settings)
        self.sequence   = sequence

        # switch off pulsing pattern 
        apply_pulsing(self.sequence, self.settings, [0], 0)


    def reset_initial(self):
        # particles to scrape -> final_dist.json
        self.partscrape = read_initial_distribution(self.settings)


    def run_simultaneous(self, sequence, turn=1):

        if not self.active:
            return 

        # collimator names 
        cnames = self.settings['scraping']['collimators']['simultaneous']['names'] 

        filename = "scrape_simultaneous"
        for cname in cnames:
            filename = filename + "_" + cname.replace(".","_")
        filename = filename + ".dat"

        pf("scraping", "simulatenous filename: " + filename)

        self.settings['survival'] = {'activate': 1, 'periodicity': 1, 'filename': filename}


        self.survival   = survival(self.settings)

        self.reset_initial()

        for cset in self.settings['scraping']['collimators']['simultaneous']['settings']:

            cname_cset = {}

            for cname in self.settings['scraping']['collimators']['simultaneous']['names']:
                cname_cset[cname] = cset

            print("cname_set", cname_cset)

            cc       = collimation(self.settings, self.settings['twiss'])
            self.settings = cc.add_collimator_settings(cname_cset)
            cc.install_all_collimator_markers(sequence, self.settings)
            sequence = cc.apply_collimator_settings(sequence, self.settings)        

            print(" ")
            tracker = xt.Tracker(_context = self.settings['context_xo'], line = sequence)
            print(" ")

            # perform the turn-by-turn tracking 
            for _ in range(1, self.settings['scraping']['turns_per_step']+1):

                turn      = turn+1
                # _turn_idx = turn-1

                tracker.track(self.partscrape, num_turns=1)

                self.survival.get_surviving_particles(turn-1, self.partscrape, extra_data={"setting": cset})

            pf("scraping", "Scraping completed with collimators {0} at {1} sigma. Surviving particles: {2}".format(self.settings['scraping']['collimators']['simultaneous']['names'], cset, self.survival.survival[-1][1]))

        self.survival.write_survival()
        
        # save the initial distribution 
        with open('final_dist_scraped_{0}_{1}_sig.json'.format(filename.replace("scrape_simultaneous", "").replace(".dat",""), cset), 'w') as fid:
            json.dump(self.partscrape.to_dict(), fid, cls=xo.JEncoder)        



    def run_independent(self, sequence, turn=1):

        if not self.active:
            return

        for cname in self.settings['scraping']['collimators']['independent'].keys():

            self.settings['survival'] = {'activate': 1, 'periodicity': 1, 
                                        'filename': 'scrape_{0}.dat'.format(cname.replace(".","_"))}

            self.survival   = survival(self.settings)

            self.reset_initial()

            for cset in self.settings['scraping']['collimators']['independent'][cname]:

                cc       = collimation(self.settings, self.settings['twiss'])
                settings = cc.add_collimator_settings({cname: cset})
                cc.install_all_collimator_markers(sequence, self.settings)
                sequence = cc.apply_collimator_settings(sequence, settings)        

                print(" ")
                tracker = xt.Tracker(_context = settings['context_xo'], line = sequence)
                print(" ")

                # perform the turn-by-turn tracking 
                for _ in range(1, settings['scraping']['turns_per_step']+1):

                    turn      = turn+1
                    # _turn_idx = turn-1

                    tracker.track(self.partscrape, num_turns=1)

                    self.survival.get_surviving_particles(turn-1, self.partscrape, extra_data={"setting": cset})

                pf("scraping", "Scraping completed with collimator {0} at {1} sigma. Surviving particles: {2}".format(cname, cset, self.survival.survival[-1][1]))

            self.survival.write_survival()
            
            # save the initial distribution 
            with open('final_dist_scraped_{0}_{1}_sig.json'.format(cname.replace(".","_"), cset), 'w') as fid:
                json.dump(self.partscrape.to_dict(), fid, cls=xo.JEncoder)        


    # def scrape(self, cname):

    #     for cset in self.settings['scraping']['collimators'][cname]:

    #         cc       = collimation(self.settings, self.settings['twiss'])
    #         settings = cc.add_collimator_settings({cname: cset})
    #         cc.install_all_collimator_markers(self.sequence, self.settings)

    #         # ## now set the collimators to their gaps 
    #         sequence = cc.apply_collimator_settings(sequence, settings)        

    #         print(" ")
    #         tracker = xt.Tracker(_context = settings['context_xo'], line = sequence)
    #         print(" ")

    #         # perform the turn-by-turn tracking 
    #         for _ in range(1, settings['scraping']['turns_per_step']+1):

    #             turn      = turn+1
    #             # _turn_idx = turn-1

    #             tracker.track(partscrape, num_turns=1)

    #             scrape.get_surviving_particles(turn-1, partscrape, extra_data={"setting": cset})

    #         pf("scraping", "Scraping completed with collimator {0} at {1} sigma. Surviving particles: {2}".format(cname, cset, scrape.survival[-1][1]))

    #     scrape.write_survival()
        
    #     # save the initial distribution 
    #     with open('final_dist_scraped_{0}_{1}_sig.json'.format(cname.replace(".","_"), cset), 'w') as fid:
    #         json.dump(partscrape.to_dict(), fid, cls=xo.JEncoder)        



    # def perform_scraping(settings, sequence, turn = 1):
        
    #     # settings to read the initial distribution 
    #     # settings['initial'] = {"filename":  "final_dist.json", 
    #     #                     "read_json": True, "read_dist": False, "create_gpdist": False, "add_closed_orbit": False }



    #     # if settings['scraping']['type'] == 'independently':

    #     for cname in settings['scraping']['collimators'].keys():

    #         settings['survival'] = {'activate': 1, 'periodicity': 1, 
    #                                 'filename': 'scrape_{0}.dat'.format(cname.replace(".","_"))}

    #         # particles to scrape -> final_dist.json
    #         partscrape = read_initial_distribution(settings)
    #         scrape     = survival(settings)

    #         # switch off pulsing pattern 
    #         apply_pulsing(sequence, settings, [0], 0)

    #         for cset in settings['scraping']['collimators'][cname]:

    #             # COLLIMATION part 
    #             # use the beta-functions to get the collimator settings 
    #             cc       = collimation(settings, settings['twiss'])

    #             settings = cc.add_collimator_settings({cname: cset})

    #             # add the upstream and downstream markers for the collimators
    #             cc.install_all_collimator_markers(sequence, settings)

    #             # ## now set the collimators to their gaps 
    #             sequence = cc.apply_collimator_settings(sequence, settings)        

    #             print(" ")
    #             tracker = xt.Tracker(_context = settings['context_xo'], line = sequence)
    #             print(" ")

    #             # perform the turn-by-turn tracking 
    #             for _ in range(1, settings['scraping']['turns_per_step']+1):

    #                 turn      = turn+1
    #                 # _turn_idx = turn-1

    #                 tracker.track(partscrape, num_turns=1)

    #                 scrape.get_surviving_particles(turn-1, partscrape, extra_data={"setting": cset})

    #             pf("scraping", "Scraping completed with collimator {0} at {1} sigma. Surviving particles: {2}".format(cname, cset, scrape.survival[-1][1]))

    #         scrape.write_survival()
            
    #         # save the initial distribution 
    #         with open('final_dist_scraped_{0}_{1}_sig.json'.format(cname.replace(".","_"), cset), 'w') as fid:
    #             json.dump(partscrape.to_dict(), fid, cls=xo.JEncoder)


        




# def perform_scraping(settings, sequence, turn = 1):
    
#     # settings to read the initial distribution 
#     settings['initial'] = {"filename":  "final_dist.json", 
#                            "read_json": True, "read_dist": False, "create_gpdist": False, "add_closed_orbit": False }



#     if settings['scraping']['type'] == 'independently':

#         for cname in settings['scraping']['collimators'].keys():

#             settings['survival'] = {'activate': 1, 'periodicity': 1, 
#                                     'filename': 'scrape_{0}.dat'.format(cname.replace(".","_"))}

#             # particles to scrape -> final_dist.json
#             partscrape = read_initial_distribution(settings)
#             scrape     = survival(settings)

#             # switch off pulsing pattern 
#             apply_pulsing(sequence, settings, [0], 0)

#             for cset in settings['scraping']['collimators'][cname]:

#                 # COLLIMATION part 
#                 # use the beta-functions to get the collimator settings 
#                 cc       = collimation(settings, settings['twiss'])

#                 settings = cc.add_collimator_settings({cname: cset})

#                 # add the upstream and downstream markers for the collimators
#                 cc.install_all_collimator_markers(sequence, settings)

#                 # ## now set the collimators to their gaps 
#                 sequence = cc.apply_collimator_settings(sequence, settings)        

#                 print(" ")
#                 tracker = xt.Tracker(_context = settings['context_xo'], line = sequence)
#                 print(" ")

#                 # perform the turn-by-turn tracking 
#                 for _ in range(1, settings['scraping']['turns_per_step']+1):

#                     turn      = turn+1
#                     # _turn_idx = turn-1

#                     tracker.track(partscrape, num_turns=1)

#                     scrape.get_surviving_particles(turn-1, partscrape, extra_data={"setting": cset})

#                 pf("scraping", "Scraping completed with collimator {0} at {1} sigma. Surviving particles: {2}".format(cname, cset, scrape.survival[-1][1]))

#             scrape.write_survival()
            
#             # save the initial distribution 
#             with open('final_dist_scraped_{0}_{1}_sig.json'.format(cname.replace(".","_"), cset), 'w') as fid:
#                 json.dump(partscrape.to_dict(), fid, cls=xo.JEncoder)


        
