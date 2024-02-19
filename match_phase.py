#%load_ext autoreload
#%autoreload 2         

import numpy as np
import xtrack as xt         
import xpart as xp
import xdeps as xd   
import inspect

path = '/afs/cern.ch/work/b/bjlindst/public/git/madx/hllhc/generate_xtrack/out/250muradXing20cmBetaNoTCLD-ORBITBUMP-TCPBETARETUNE-OFFSETAPERir7re12c6/'
#path = '/afs/cern.ch/work/b/bjlindst/public/git/madx/hllhc/ir7_match/'
lhc = xt.line.Line.from_json(path+"b4_sequence_noaper.json")
#lhcs = xt.Multiline.from_json(path+'hllhc.json')
#lhc = lhcs.lhcb2

mt_cry1 = xt.Marker()
s_position = lhc.get_length() - 6775
lhc.insert_element('mt_cry1',element=mt_cry1, at_s=s_position)   
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

##### initialize some twiss
tw = lhc.twiss()

bir3b2 = tw.get_twiss_init("s.ds.l3.b2")
eir3b2 = tw.get_twiss_init("e.ds.r3.b2")    
btcp7b2 = tw.get_twiss_init("tcp.d6r7.b2") 
tw.rows[['ip1.l1','tcp.d6r7.b2','mt_cry1','ip1']].cols[['s','mux','muy','betx','bety']]
##### function to get phase advance 
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
    lhc, ele_start="tcp.d6r7.b2", ele_stop="mt_cry1", twinit=btcp7b2
)        
TPhase = xt.TargetRelPhaseAdvance
class FractionalPhase (TPhase):

    def compute(self, tw):
        out = TPhase.compute(self, tw)
        out = out % 1
        return out  

optphase = 158.521582158919/360

  
opt = lhc.match(solve=False,
                default_tol={None: 5e-8},
                solver_options=dict(max_rel_penalty_increase=2.),
                #twiss_init='preserve_start',
                #table_for_twiss_init=(tw),
                #ele_start=('s.ds.l7.b2'),
                #ele_stop=('mt_cry1'),
                method='4d', # <- passed to twiss
                vary=[
                    xt.VaryList(['kqtf.b2', 'kqtd.b2'], step=1e-8, tag='quad'),
                    xt.VaryList(['ksf.b2', 'ksd.b2'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
                    xt.VaryList(list(strDictIR2.keys()), tag='ir2'), 
                    xt.VaryList(list(strDictIR4.keys()), tag='ir4')
                ],
                targets = [
                    xt.TargetSet(qx=62.313, qy=60.318, tol=1e-6, tag='tune'),
                    xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
                    #xt.Target(action=act_pa2, tar='muy_mod', value=optphase, tol=1e-3, tag='ph'),
                    FractionalPhase('muy', optphase, tol=1e-3, end='mt_cry1', start='tcp.d6r7.b2', tag='ph'),
                    xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at='s.ds.l4.b2', tag='ir4'),
                    #xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at='s.ds.r4.b2', tag='ir4'),
                    xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at='s.ds.l2.b2', tag='ir2'),
                    #xt.TargetSet(['alfx','alfy','betx','bety','dx','dpx'],value=tw,at='s.ds.r2.b2', tag='ir2')  
                ]                                             
)
opt.assert_within_tol=False

# first set tune and chroma correct
opt.disable_targets(tag='ph')
opt.disable_targets(tag='ir2')
opt.disable_targets(tag='ir4')
opt.disable_vary(tag='ir2')
opt.disable_vary(tag='ir4')
doMatch(opt)

opt.enable_targets(tag='ph')
opt.enable_targets(tag='ir2')
opt.enable_targets(tag='ir4')
opt.enable_vary(tag='ir2')
opt.enable_vary(tag='ir4')

doMatch(opt)




