

# Load required packages
import numpy as np
import time, yaml, sys, json, sys, sqlite3
import matplotlib.pyplot as plt

# xt_dev = True
# if xt_dev:
#     sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xtrack")
#     sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xpart")
#     sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xdeps")
#     sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xobjects")
#     sys.path.insert(0, "/afs/cern.ch/work/p/pahermes/private/xsuite/dev/xfields")


import xobjects as xo
import xpart as xp
import pandas as pd
import xtrack as xt

def element_properties(name):
    _idx = sequence.element_names.index(name)
    return sequence.elements[_idx].to_dict()

sys.path.append("/afs/cern.ch/user/p/pahermes/public/simcontrol/templates/xtrack/")
sys.path.append("/afs/cern.ch/user/p/pahermes/public/simcontrol/python_scripts/")

import xtrack_tools as xtt

# ## Load settings and inputs
# load the settings file 
settings = xtt.load_settings()



## ADT WHITE NOISE TIME PROFILE 

# generate pulsing pattern 
# adt_kick = (np.random.uniform(low=0.0, high=1.0, size=settings['simulation']['nturns'])-0.5)/0.5

gain_ramp_time       = settings['pulsing_pattern']['ramp_adt']['gain_ramp_time']
gain_ramp_time       = int(gain_ramp_time)

rampdown_final_level = settings['pulsing_pattern']['ramp_adt']['rampdown_final_level']
rampdown_final_level = float(rampdown_final_level)

adt_kick             = xtt.generate_adt_time_profile(settings, gain_ramp_time, rampdown_final_level)





# get the to_np function depending on the context 
to_np    = xtt.get_to_np(settings)

# load sequence
sequence = xtt.sequence.load_sequence(settings)



# set cavity properties 
for cav in sequence.get_elements_of_type(xt.Cavity)[1]:
    idx = sequence.element_names.index(cav)   
    sequence.elements[idx].voltage   = 1500000.0
    sequence.elements[idx].frequency = 400789598.98582596
    sequence.elements[idx].lag       = 180.0

# # get twiss parameters
tw        = xtt.get_twiss(settings, sequence.copy())

# settings['initial']['filename'] = "/afs/cern.ch/work/p/pahermes/private/401_quenchtest22/000_initial_dist/initial.dgauss.run3.dat"

# read initial distribution 
part      = xtt.read_initial_distribution(settings)

# set the elements to their reference values 
# apply settings specified in "set_elements"
xtt.set_element_value(sequence, settings)

# COLLIMATION part 
# use the beta-functions to get the collimator settings 
cc       = xtt.collimation(settings, tw)
settings = cc.add_collimator_settings()

# add the upstream and downstream markers for the collimators
cc.install_all_collimator_markers(sequence, settings)

## now set the collimators to their gaps 
cc.apply_collimator_settings(sequence, settings)

# get the reference settings for all elements of the sequence that can be pulsed
xtt.add_all_reference_settings(sequence, settings)

xtt.save_tracked_sequence(sequence)


# # Prepare tracking 
## Transfer lattice on context and compile tracking code
tracker = xt.Tracker(_context = settings['context_xo'], line = sequence)

# get ready for the initial distribution 
with open("dist0.json", 'r') as fid:
    part= xp.Particles.from_dict(json.load(fid), _context=settings['context_xo'])

survival  = xtt.survival(settings)
emittance = xtt.emittance(settings)
dump      = xtt.dump(settings)


turn_start = settings['simulation']['turn_start']
turn_end   = settings['simulation']['turn_start'] + settings['simulation']['nturns']


# dump the starting conditions 
dump.dump_turn(part, turn_start-1, to_np)

# perform the turn-by-turn tracking 
for turn in range(1,settings['simulation']['nturns']+1):
    
    _turn_idx = turn-1
    
    if turn % 10 == 0:
        print("Tracking turn: {0}".format(turn))
    # apply the pulsing pattern 
    xtt.apply_pulsing(tracker.line, settings, adt_kick, turn)
    
    tracker.track(part, num_turns=1)
    
    survival.get_surviving_particles(turn, part)
    emittance.get_emittance(turn, part)
    dump.dump_turn(part, turn, to_np)

survival.write_survival()
emittance.write_emittance_table()
xtt.collimation.write_coll_summary(part, sequence, settings)

dump.write_dumpfile()

# save the initial distribution 
with open('final_dist.json', 'w') as fid:
    json.dump(part.to_dict(), fid, cls=xo.JEncoder)

xtt.save_settings(settings)











