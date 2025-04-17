#%load_ext autoreload
#%autoreload 2         

import numpy as np
import xtrack as xt         
import xpart as xp
import xdeps as xd   
import numpy as np
import inspect
import pickle
import sys
import os
from IPython import embed



def main():

    optphase_deg  = 150 #float(sys.argv[1])
    deg_name = round(optphase_deg)
    optphase = optphase_deg/360

    print(f'Phase to achieve: {optphase_deg} deg')

    
    line_path_noaper = '../track_flat_top_b2_no_aper.json'
    #line_path_noaper = '../track_flat_top_b2_no_aper_THICK.json'
    line_path_aper = 'track_flat_top_b2.json'

    print('Loading line.....  ', line_path_noaper)

    lhc = xt.line.Line.from_json(line_path_noaper)  #"b4_sequence_noaper.json"

    TCCS_name = 'tccs.5r3.b2'
    TCP_name = 'tcp.d6r7.b2'

    TCCS_loc_abs  = 6773.9428  #6773.7 #6775

    mt_cry1 = xt.Marker()
    s_position = lhc.get_length() - TCCS_loc_abs
    lhc.insert_element(TCCS_name, element=mt_cry1, at_s=s_position)   
    lhc.build_tracker()   
    #lhc.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=6800e9)
    lhc.twiss_default['method'] = '4d' 



    tw = lhc.twiss()

    ##### helper functions
    def doMatch(opt):
        opt.solve()
        matchSummary(opt)
        opt.log().rows[-5:]  

    def matchSummary(opt):
        for tt in opt._err.targets:
            if tt.line:
                nn=" ".join((tt.line,)+tt.tar)
                rr=tt.action.run()[tt.line][tt.tar]
            else:
                nn=tt.tar
                if(callable(nn)):
                    temptw = lhc.twiss()
                    rr = nn(temptw)
                    del temptw  
                    string = nn.__repr__()
                    str1 = string.split('(')[1]
                    str2 = string.split('(')[2].split(')')[0]
                    nn = [str1,str2]
                else:
                    rr=tt.action.run()[tt.tar]
                nn=": ".join(nn)
            if(type(tt.value)==xt.match.LessThan):
                vv=tt.value.upper
                dd=(rr-vv)
                print(f'{nn:25}: {rr:15.7e} {vv:15.7e} d={dd:15.7e} {rr<(vv+tt.tol)}') 
            elif(type(tt.value)==xt.match.GreaterThan):
                vv=tt.value.lower
                dd=(rr-vv)
                print(f'{nn:25}: {rr:15.7e} {vv:15.7e} d={dd:15.7e} {rr>(vv-tt.tol)}')
            elif(hasattr(tt,'rhs')):
                vv=tt.rhs
                dd=(rr-vv)
                if(tt.ineq_sign=='>'):
                    print(f'{nn:25}: {rr:15.7e} {vv:15.7e} d={dd:15.7e} {rr>(vv-tt.tol)}')
                else:
                    print(f'{nn:25}: {rr:15.7e} {vv:15.7e} d={dd:15.7e} {rr<(vv+tt.tol)}')   
            else:
                vv=tt.value
                dd=(rr-vv)
                dd=np.abs(dd)
                print(f'{nn:25}: {rr:15.7e} {vv:15.7e} d={dd:15.7e} {dd<tt.tol}') 




    ##### function to get phase advance 

    def getPhaseAdvance_muy_fractional(tw, start_name, end_name):
        return (tw['muy', end_name] - tw['muy', start_name]) % 1
    def getPhaseAdvance(tw, start_name, end_name):
        return (tw['muy', end_name] - tw['muy', start_name])
    def getPhaseAdvance_deg(tw, start_name, end_name):
        return (tw['muy', end_name] - tw['muy', start_name]) % 1 * 360

    MQTLI_l4 = 'mqtli.11l4.b2'
    MQTLI_r2 = 'mqtli.11r2.b2'
    MQTLI_l3 = 'mqtli.11l3.b2'
    MQTLI_r3 = 'mqtli.11r3.b2'
  

    phaseTCP_cry0 = getPhaseAdvance(tw, TCP_name, TCCS_name)
    phase_start_end = getPhaseAdvance(tw, MQTLI_l4, MQTLI_r2) # MUST BE CONSERVED

    new_tcp_cry_phase = int(phaseTCP_cry0) + optphase
    print("Origunal phase TCP to Cry: ", phaseTCP_cry0, " New phase TCP to Cry: ", new_tcp_cry_phase)
    print("Phase to preserve: ", phase_start_end)
   
    # ??? LIMITS
    knobs_lss2_r = ['kqtl11.r2b2', 'kqt12.r2b2', 'kqt13.r2b2']
    knobs_lss3_l = ['kqtl11.l3b2', 'kqt12.l3b2', 'kqt13.l3b2']
    knobs_a23 = ['kqf.a23', 'kqd.a23']
    knobs_lss3_r = ['kqtl11.r3b2', 'kqt12.r3b2', 'kqt13.r3b2']
    knobs_a34 = ['kqf.a34', 'kqd.a34']
    knobs_lss4_l = ['kqtl11.l4b2', 'kqt12.l4b2', 'kqt13.l4b2']


    opt = lhc.match(solve=False,
                    default_tol={None: 5e-7},  #{None: 5e-8}
                    solver_options={"max_rel_penalty_increase": 2.}, 
                    method='4d',
                    vary=[
                        xt.VaryList(knobs_lss2_r, step=1e-6, tag='quad'),                      
                        xt.VaryList(knobs_lss3_l, step=1e-6, tag='quad'),                      
                        xt.VaryList(knobs_a23, step=1e-8, tag='quad'),                      
                        xt.VaryList(knobs_lss3_r, step=1e-6, tag='quad'),                      
                        xt.VaryList(knobs_a34, step=1e-8, tag='quad'),                      
                        xt.VaryList(knobs_lss4_l, step=1e-6, tag='quad'),                                  
                    ],
                    targets = [
                        xt.TargetSet(qx=62.28, qy=60.31, tol=1e-6, tag='tune'),       # TRY WITH TUNE                     
                        xt.TargetRelPhaseAdvance('muy', new_tcp_cry_phase, tol=1e-3, end = TCCS_name, start = TCP_name, tag='ph_tcp_cry'),
                        xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at=MQTLI_l4, tag='MQTLI_l4'),
                        xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at=MQTLI_r2, tag='MQTLI_r2'),
                        #xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at=MQTLI_l3, tag='MQTLI_l3'),
                        #xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at=MQTLI_r3, tag='MQTLI_r3')
                    ]                                             
    )
    opt.assert_within_tol=False

 
    doMatch(opt)
    tw = lhc.twiss(method='4d')
    print(f"Phase adv: {getPhaseAdvance_deg(tw, TCP_name, TCCS_name)}")
    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")

    embed()


    # ----------------------- Save the knobs and the line -----------------------
    knobs = opt.get_knob_values()

    print('knob values: \n', knobs)

    knob_file = f'./knobs_{deg_name}_new_test.pkl'
    with open(knob_file, 'wb') as f:
        pickle.dump(knobs, f)

    print('\n----------------------------------------------------------------------')
    print('\nLoading line with apertures.....  ', line_path_aper)

    line = xt.Line.from_json(line_path_aper)
    
    for k, v in knobs.items():
        line.vars[k] = v


    
    opt2 = line.match(solve=False,
                    default_tol={None: 5e-8},  #{None: 5e-8}
                    solver_options={"max_rel_penalty_increase": 2.}, 
                    #method='4d',
                    vary=[
                        xt.VaryList(['kqtf', 'kqtd'], step=1e-8, tag='quad'),                      
                        xt.VaryList(['ksf', 'ksd'], step=1e-4,  tag='sext'),      #limits=[-0.1, 0.1],
                    ],
                    targets = [
                        xt.TargetSet(qx=62.28, qy=60.31, tol=1e-6, tag='tune'),                         
                        xt.TargetSet(dqx=10.0, dqy=10.0, tol=0.01, tag='chrom'),                           
                    ]                                             
    )
    opt2.assert_within_tol=False

    
    print('Re-matching tune and chroma.....')
    doMatch(opt2)
    print('knob values: \n', opt2.get_knob_values())


    # ----------------------- Save line with knobs -----------------------
    print('\n----------------------------------------------------------------------')
    new_line_name = f'flat_top_b2_phadv_{deg_name}_new_test_nice2.json'
    print('Saving line.....  ', new_line_name)
    line.to_json(new_line_name)

    print("Check parameters:  \n")
    line.discard_tracker()
    end_s = line.get_length()
    TCCS_loc = end_s - TCCS_loc_abs #6773.7
    line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)
    tw = line.twiss(method='4d')
    print(f"Phase adv: {getPhaseAdvance_deg(tw, TCP_name, TCCS_name)}")
    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")




    # ----------------------- Check new line  -----------------------
    print('\n----------------------------------------------------------------------')

    print('Loading line.....  ', new_line_name)
    new_line = xt.Line.from_json(new_line_name)

    end_s = new_line.get_length()

    TCCS_loc = end_s - TCCS_loc_abs #6773.7
    new_line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)

    #tw = new_line.twiss(method='4d')
    tw = new_line.twiss()   
    print(f"Phase adv: { (float(tw['muy', TCCS_name])% 1* 2*np.pi - float(tw['muy', TCP_name])% 1* 2*np.pi)*180/np.pi}")
    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")

    print(f"Phase adv to achieve:{optphase},\t  Phase adv set: { float(tw['muy', TCCS_name])% 1 - float(tw['muy', TCP_name])% 1}")







if __name__ == "__main__":
    main()

