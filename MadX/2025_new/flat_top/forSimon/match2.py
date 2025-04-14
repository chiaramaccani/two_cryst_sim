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

    lhc = xt.line.Line.from_json('flat_top_b2_phadv_150_new_test.json')  
    end_s = lhc.get_length()

    TCCS_name = 'tccs.5r3.b2'
    TCP_name = 'tcp.d6r7.b2'

    TCCS_loc_abs  = 6773.9428  #6773.7 #6775
    TCCS_loc = end_s - TCCS_loc_abs #6773.7


    lhc.insert_element(at_s=TCCS_loc, element=xt.Marker(), name=TCCS_name)

    tw = lhc.twiss(method='4d')
    df = tw.to_pandas()
    df0 = df[(df['name']==TCCS_name) |  (df['name']==TCP_name)]# TCP_name]] 


    print(f"Phase adv: { (float(tw['muy', TCCS_name])% 1* 2*np.pi - float(tw['muy', TCP_name])% 1* 2*np.pi)*180/np.pi}")

    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")


    tw = lhc.twiss()
    df = tw.to_pandas()
    df0 = df[(df['name']==TCCS_name) |  (df['name']==TCP_name)]# TCP_name]] 


    print(f"Phase adv: { (float(tw['muy', TCCS_name])% 1* 2*np.pi - float(tw['muy', TCP_name])% 1* 2*np.pi)*180/np.pi}")

    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")



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


    opt = lhc.match(solve=False,
                    default_tol={None: 5e-7},  #{None: 5e-8}
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
    opt.assert_within_tol=False



    print('Matching tune and chroma.....')
    doMatch(opt)


    tw = lhc.twiss(method='4d')
    df = tw.to_pandas()
    df0 = df[(df['name']==TCCS_name) |  (df['name']==TCP_name)]# TCP_name]] 


    print(f"Phase adv: { (float(tw['muy', TCCS_name])% 1* 2*np.pi - float(tw['muy', TCP_name])% 1* 2*np.pi)*180/np.pi}")

    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")


    tw = lhc.twiss()
    df = tw.to_pandas()
    df0 = df[(df['name']==TCCS_name) |  (df['name']==TCP_name)]# TCP_name]] 


    print(f"Phase adv: { (float(tw['muy', TCCS_name])% 1* 2*np.pi - float(tw['muy', TCP_name])% 1* 2*np.pi)*180/np.pi}")

    print(f"Tune: {tw['qx']}, {tw['qy']}")
    print(f"Chrom: {tw['dqx']}, {tw['dqy']}")

    embed()

if __name__ == "__main__":
    main()
