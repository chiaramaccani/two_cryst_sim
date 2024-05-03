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



def main():


    path = './'
    lhc = xt.line.Line.from_json(path+"b1_sequence_noaper.json")  #"b4_sequence_noaper.json"


    lhc.build_tracker() 
    lhc.particle_ref = xt.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)
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
    nrj = 7000
    scale = 23348.89927
    scmin = 0.03*7000./nrj
    qtlimitx28 = 1.0*225.0/scale
    qtlimitx15 = 1.0*205.0/scale
    qtlimit2 = 1.0*160.0/scale
    qtlimit3 = 1.0*200.0/scale
    qtlimit4 = 1.0*125.0/scale
    qtlimit5 = 1.0*120.0/scale
    qtlimit6 = 1.0*90.0/scale   

    ### lhc.vars['kqtf.b2'] = 1e-8  # set knob to certain value
    ##### initialize some twiss
    tw = lhc.twiss()

    """bir3b2 = tw.get_twiss_init("s.ds.l3.b2")
    eir3b2 = tw.get_twiss_init("e.ds.r3.b2")    
    btcp7b2 = tw.get_twiss_init("tcp.d6r7.b2") 
    tw.rows[['ip1.l1','tcp.d6r7.b2','mt_cry1','ip1']].cols[['s','mux','muy','betx','bety']]"""
  



    
    opt = lhc.match(solve=False,
                    default_tol={None: 5e-8},
                    solver_options={"max_rel_penalty_increase": 2.}, 
                    method='4d', # <- passed to twiss
                    vary=[
                        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, tag='quad'),                        
                        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),        
                    ],
                    targets = [
                        xt.TargetSet(qx=62.313, qy=60.318, tol=1e-6, tag='tune'),                         
                        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),                          
                    ]                                             
    )
    opt.assert_within_tol=False

    # first set tune and chroma correct
    doMatch(opt)


    doMatch(opt)
    doMatch(opt)







    # ----------------------- Save the knobs and the line patched -----------------------
    #lhc.to_json('HL_phase_noaper.json')
    print('\n----------------------------------------------------------------------')
    knobs = opt.get_knob_values()

    print('knob values: \n', knobs)

    knob_file = f'./knobs.pkl'
    with open(knob_file, 'wb') as f:
        pickle.dump(knobs, f)


    line_path = './b1_sequence.json'  #b4_sequence_patched
    line = xt.Line.from_json(line_path)

    for k, v in knobs.items():
        line.vars[k] = v


    # ----------------------- Save line with knobs -----------------------
    print('\n----------------------------------------------------------------------')
    new_line_name = f'b1_sequence_tune.json'
    line.to_json(new_line_name)




    # ----------------------- Check new line  -----------------------

    print('Loading line.....  ', new_line_name)
    new_line = xt.Line.from_json(new_line_name)


    tw = new_line.twiss(method='4d')
    df = tw.to_pandas()
    print(df[['mux', 'muy']].iloc[-1])






if __name__ == "__main__":
    main()
