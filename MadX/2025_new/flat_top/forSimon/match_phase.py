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

    
    #line_path_noaper = 'track_flat_top_b2_no_aper.json'
    line_path_noaper = '../track_flat_top_b2_no_aper_THICK.json'
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


    ##### use ir2 and ir4 for changing the phase, knobs:
    nrj = 6800
    scale = 23348.89927
    scmin = 0.03*7000./nrj
    qtlimitx28 = 1.0*225.0/scale
    qtlimitx15 = 1.0*205.0/scale
    qtlimit2 = 1.0*160.0/scale
    qtlimit3 = 1.0*200.0/scale
    qtlimit4 = 1.0*125.0/scale
    qtlimit5 = 1.0*120.0/scale
    qtlimit6 = 1.0*90.0/scale   

    strDictIR2 = {
        'kqt13.l2b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
        'kqt12.l2b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
        'kqtl11.l2b2' :{'step': 1e-6, 'limits':  [-qtlimit4,qtlimit4],},
        'kq10.l2b2'   :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq9.l2b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq8.l2b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq7.l2b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq6.l2b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq5.l2b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq4.l2b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kqx.l2'      :{'step': 1e-6, 'limits':  [-qtlimitx28,qtlimitx28],},
        'kq4.r2b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq5.r2b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq6.r2b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq7.r2b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq8.r2b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq9.r2b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq10.r2b2'   :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kqtl11.r2b2' :{'step': 1e-6, 'limits':  [-qtlimit4,qtlimit4],},
        'kqt12.r2b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
        'kqt13.r2b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
    }

    strDictIR4 = {
        'kqt13.l4b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
        'kqt12.l4b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
        'kqtl11.l4b2' :{'step': 1e-6, 'limits':  [-qtlimit4,qtlimit4],},
        'kq10.l4b2'   :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq9.l4b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq8.l4b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq7.l4b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq6.l4b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq5.l4b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq5.r4b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq6.r4b2'    :{'step': 1e-6, 'limits':  [-qtlimit2,qtlimit2],},
        'kq7.r4b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq8.r4b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq9.r4b2'    :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kq10.r4b2'   :{'step': 1e-6, 'limits':  [-qtlimit3,qtlimit3],},
        'kqtl11.r4b2' :{'step': 1e-6, 'limits':  [-qtlimit4,qtlimit4],},
        'kqt12.r4b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
        'kqt13.r4b2'  :{'step': 1e-6, 'limits':  [-qtlimit5,qtlimit5],},
    }       
        
    lhc.vars.vary_default.update(strDictIR2) 
    lhc.vars.vary_default.update(strDictIR4)

    tw = lhc.twiss()

    bir3b2 = tw.get_twiss_init("s.ds.l3.b2")
    eir3b2 = tw.get_twiss_init("e.ds.r3.b2")    
    btcp7b2 = tw.get_twiss_init("tcp.d6r7.b2") 
    tw.rows[['ip1.l1','tcp.d6r7.b2',TCCS_name,'ip1']].cols[['s','mux','muy','betx','bety']]





    ##### function to get phase advance 

    def getPhaseAdvance_muy_fractional(tw, start_name, end_name):
        return (tw['muy', end_name] - tw['muy', start_name]) % 1
    def getPhaseAdvance(tw, start_name, end_name):
        return (tw['muy', end_name] - tw['muy', start_name])

    class phaseAdvance(xd.Action):
        def __init__(self, line, ele_start, ele_stop, twinit):
            self.line = line
            self.ele_start = ele_start
            self.ele_stop = ele_stop
            self.twinit = twinit
            
        def run(self):
            twObj = self.line.twiss(start='tcp.d6r7.b2', end=self.ele_stop, init=self.twinit)
            mux1 = twObj.rows[self.ele_start].cols['mux']['mux'][0]
            mux2 = twObj.rows[self.ele_stop].cols['mux']['mux'][0]
            muy1 = twObj.rows[self.ele_start].cols['muy']['muy'][0]
            muy2 = twObj.rows[self.ele_stop].cols['muy']['muy'][0]
            del twObj
            deltaMux = mux2-mux1
            deltaMuy = muy2-muy1

            return {'mux': deltaMux, 'muy': deltaMuy, 'mux_mod': deltaMux%1, 'muy_mod': deltaMuy%1}  

    act_pa2 = phaseAdvance(
        lhc, ele_start="tcp.d6r7.b2", ele_stop=TCCS_name, twinit=btcp7b2
    )        
    TPhase = xt.TargetRelPhaseAdvance
    class FractionalPhase (TPhase):

        def compute(self, tw):
            out = TPhase.compute(self, tw)
            out = out % 1
            return out  


    """phaseTCP_IP2 = getPhaseAdvance_muy_fractional(tw, TCP_name, 'ip2')
    phaseTCP_cry0 = getPhaseAdvance_muy_fractional(tw, TCP_name, TCCS_name)
    phaseCryIP20 = getPhaseAdvance_muy_fractional(tw, TCCS_name, 'ip2')"""

    phaseTCP_IP2 = getPhaseAdvance(tw, TCP_name, 'ip2')
    phaseTCP_cry0 = getPhaseAdvance(tw, TCP_name, TCCS_name)
    phaseCryIP20 = getPhaseAdvance(tw, TCCS_name, 'ip2')

    print(f"\nPhase TCP to Cry:     {phaseTCP_cry0}, Phase Cry to IP2:     {phaseCryIP20}, sum: {phaseTCP_cry0 + phaseCryIP20}, Phase TCP to IP2: {phaseTCP_IP2},  ")

    new_tcp_cry_phase = int(phaseTCP_cry0) + optphase
    new_cry_ip2_phase =    phaseTCP_IP2 - new_tcp_cry_phase

    print(f"New phase TCP to Cry: {new_tcp_cry_phase}, New phase Cry to IP2: {new_cry_ip2_phase}, sum {new_tcp_cry_phase + new_cry_ip2_phase}, Phase TCP to IP2: {phaseTCP_IP2},")

    opt = lhc.match(solve=False,
                    default_tol={None: 5e-7},  #{None: 5e-8}
                    solver_options={"max_rel_penalty_increase": 2.}, 
                    method='4d',
                    vary=[
                        xt.VaryList(['kqtf', 'kqtd'], step=1e-8, tag='quad'),                      
                        xt.VaryList(['ksf', 'ksd'], step=1e-4,  tag='sext'),      #limits=[-0.1, 0.1],
                        xt.VaryList(list(strDictIR2.keys()), tag='ir2'), 
                        xt.VaryList(list(strDictIR4.keys()), tag='ir4')
                    ],
                    targets = [
                        xt.TargetSet(qx=62.28, qy=60.31, tol=1e-6, tag='tune'),                         
                        xt.TargetSet(dqx=10.0, dqy=10.0, tol=0.01, tag='chrom'),                           
                        #FractionalPhase('muy', optphase, tol=1e-3, end = TCCS_name, start = TCP_name, tag='ph_tcp_cry'),
                        #FractionalPhase('muy',  phaseTCP_IP2 - optphase, tol=1e-3, start = TCCS_name, end='ip2', tag='ph_cry_IP2'), #phaseCryIP20+(phaseTCP_cry0-optphase)
                        xt.TargetRelPhaseAdvance('muy', new_tcp_cry_phase, tol=1e-3, end = TCCS_name, start = TCP_name, tag='ph_tcp_cry'),
                        xt.TargetRelPhaseAdvance('muy', new_cry_ip2_phase, tol=1e-3, start = TCCS_name, end='ip2', tag='ph_cry_IP2'), #phaseCryIP20+(phaseTCP_cry0-optphase)
                        xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at='s.ds.l4.b2', tag='ir4'),
                        xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at='ip2', tag='ip2'),
                        xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at='s.ds.l2.b2', tag='ir2')
                    ]                                             
    )
    opt.assert_within_tol=False



    print('\nMatching tune and chroma.....')
    opt.disable_targets(tag='ph_tcp_cry')
    opt.disable_targets(tag='ph_cry_IP2')
    opt.disable_targets(tag='ir2')
    opt.disable_targets(tag='ir4')
    opt.disable_vary(tag='ir2')
    opt.disable_vary(tag='ir4')
    doMatch(opt)
    opt.disable_targets(tag='tune')
    opt.disable_targets(tag='chrom')
    opt.disable_vary(tag='quad')
    opt.disable_vary(tag='sext')

    """print('\nMatching phase TCP to Cry.....')
    opt.enable_targets(tag='ph_tcp_cry')
    opt.enable_targets(tag='ir4')
    opt.enable_vary(tag='ir4')
    doMatch(opt)
    opt.disable_targets(tag='ph_tcp_cry')
    opt.disable_targets(tag='ir4')
    opt.disable_vary(tag='ir4')

    print('\nMatching phase Cry to IP2.....')
    opt.enable_targets(tag='ph_cry_IP2')
    opt.enable_targets(tag='ir2')
    opt.enable_vary(tag='ir2')
    doMatch(opt)

    print("\nEnabling all targets and vary.....")
    opt.enable_all_targets()
    opt.enable_all_vary()
    doMatch(opt)"""


    """opt.disable_targets(tag='ph_tcp_cry')
    opt.disable_targets(tag='ph_cry_IP2')
    opt.disable_targets(tag='ir2')
    opt.disable_targets(tag='ir4')
    opt.disable_vary(tag='ir2')
    opt.disable_vary(tag='ir4')
    doMatch(opt)

    opt.disable_targets(tag='tune')
    opt.disable_targets(tag='chrom')
    opt.disable_vary(tag='quad')
    opt.disable_vary(tag='sext')
    opt.enable_targets(tag='ph_tcp_cry')
    opt.enable_targets(tag='ph_cry_IP2')
    opt.enable_targets(tag='ir2')
    opt.enable_targets(tag='ir4')
    opt.enable_vary(tag='ir2')
    opt.enable_vary(tag='ir4')
    doMatch(opt)
    doMatch(opt)
    """

    ### EVENTUALLY DO A FINAL MATCH WITH OPT.ENABLE_ALL_TARGETS() AND OPT.ENABLE_ALL_VARY()
    ### another option, after matching tune and chroma the first time, disable IR2 vary and start by matching phase TCP to Cry,\
    ### then match phase Cry to IP2 (with IR4 vary disable)







    # ----------------------- Save the knobs and the line -----------------------
    #lhc.to_json('HL_phase_noaper.json')
    print('\n----------------------------------------------------------------------')
    knobs = opt.get_knob_values()

    print('knob values: \n', knobs)

    """knob_file = f'./knobs_db/knobs_{deg_name}_new_test.pkl'
    with open(knob_file, 'wb') as f:
        pickle.dump(knobs, f)"""

    print('\nLoading line with apertures.....  ', line_path_aper)

    line = xt.Line.from_json(line_path_aper)
    end_s = line.get_length()

    TCCS_loc = end_s - TCCS_loc_abs #6773.7
    line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)

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



    print('Matching tune and chroma.....')
    doMatch(opt2)


    tw = line.twiss()
    df = tw.to_pandas()
    df0 = df[(df['name']==TCCS_name) |  (df['name']==TCP_name)]# TCP_name]] 


    print(f"Phase adv: { (float(tw['muy', TCCS_name])% 1* 2*np.pi - float(tw['muy', TCP_name])% 1* 2*np.pi)*180/np.pi}")

    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")


    tw = line.twiss()
    df = tw.to_pandas()
    df0 = df[(df['name']==TCCS_name) |  (df['name']==TCP_name)]# TCP_name]] 


    # ----------------------- Save line with knobs -----------------------
    print('\n----------------------------------------------------------------------')
    new_line_name = f'flat_top_b2_phadv_{deg_name}_new_test.json'
    line.to_json(new_line_name)




    # ----------------------- Check new line  -----------------------

    print('Loading line.....  ', new_line_name)
    new_line = xt.Line.from_json(new_line_name)

    end_s = new_line.get_length()

    TCCS_loc = end_s - TCCS_loc_abs #6773.7
    new_line.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)

    #tw = new_line.twiss(method='4d')
    tw = new_line.twiss()
    df = tw.to_pandas()
    embed()
    df0 = df[(df['name']==TCCS_name) |  (df['name']==TCP_name)]# TCP_name]] 

    print(f"Phase adv to achieve:{optphase},\t  Phase adv set: {(df0['muy'].iloc[1] - df0['muy'].iloc[0])%1}")
    print(f"Phase adv: { (float(tw['muy', TCCS_name])% 1* 2*np.pi - float(tw['muy', TCP_name])% 1* 2*np.pi)*180/np.pi}")

    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")







if __name__ == "__main__":
    main()


"""
Message from root@lxplus929.cern.ch on <no tty> at 16:06 ...
Dear LxPlus User cmaccani

The process (python) has been killed as the CGroup of all
your processes has reached your memory allocation.

Memory cgroup out of memory: Killed process 2216162 (python) total-vm:38462348kB, anon-rss:35904008kB, file-rss:6272kB, shmem-rss:0kB, UID:150424 pgtables:70672kB oom_score_adj:0

To check your memory limits of your CGroup and current usage:

systemctl status user-0.slice
EOF

"""