
def calcAction(p0,tw,ele,exn=2.5e-6,nrj=7000e9,debug=False):
    ex = exn*0.938e9/nrj
    alfx = tw.rows[ele].cols['alfx'].alfx[0]
    alfy = tw.rows[ele].cols['alfy'].alfy[0]
    betx = tw.rows[ele].cols['betx'].betx[0]
    bety = tw.rows[ele].cols['bety'].bety[0]
    xx0 = tw.rows[ele].cols['x'].x[0]
    yy0 = tw.rows[ele].cols['y'].y[0]
    pxx0 = tw.rows[ele].cols['px'].px[0]
    pyy0 = tw.rows[ele].cols['py'].py[0]
    x0 = p0.x - xx0
    y0 = p0.y - yy0
    px0 = p0.px - pxx0
    py0 = p0.py - pyy0
    jx = np.sqrt(x0**2/betx + (alfx*x0/np.sqrt(betx) + np.sqrt(betx)*px0)**2)/np.sqrt(ex)
    jy = np.sqrt(y0**2/bety + (alfy*y0/np.sqrt(bety) + np.sqrt(bety)*py0)**2)/np.sqrt(ex)

    if debug:
        print(tw.rows[ele].cols[['alfx','alfy','betx','bety','x','y','px','py']])
    return jx,jy

def plotHist(p1,p2,figdim=(20,10), nbins=200, save=False, savePath="plot", line = None, density=False,scale=1, range=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,figsize=figdim)
    ax.hist([p1*scale,p2*scale],nbins,density=density,color=['red','green'])
    if line is not None:
        if not hasattr(line, '__iter__') or isinstance(line, str):
            line = [line]
        for l in line:
            ax.axvline(x=l, color='black', linestyle='--')

    if range is not None:
        ax.set_xlim(range)
    plt.show()

#coll_manager0.disable_scattering()
#part = coll_manager0.generate_pencil_on_collimator(tcp, num_particles=1e7, impact_parameter=2e-6)

idx_TCP = line0.element_names.index(tcp)
idx_TARGET = line0.element_names.index(TARGET_name)

tw0 = line0.twiss()
tw1 = line1.twiss()

coll_manager0.enable_scattering()
coll_manager1.enable_scattering()

part_before_TCP = part.copy()
#line.element_dict['tcp.d6r7.b2'].track(part)


line0.track(part, num_turns=1, ele_stop='ip6')
jxT,jyT = calcAction(part,tw0,'tcp.d6r7.b2')
m1 = jyT>7.2
part_ip6 = part.filter(m1)
part_TCCS_10 = part_ip6.copy() # for tracking without rematched phase
part_TCCS_11 = part_ip6.copy() # for tracking with rematched phase
line0.track(part_TCCS_10, num_turns=1, ele_start='ip6', ele_stop=TCCS_name)
line1.track(part_TCCS_11, num_turns=1, ele_start='ip6', ele_stop=TCCS_name)

idx_TCCS = line0.element_names.index(TCCS_name)
ele2 = line0.element_names[idx_TCCS+2]

part_afterTCCS_20 = part_TCCS_10.copy()
part_afterTCCS_21 = part_TCCS_11.copy()
line0.track(part_afterTCCS_20, num_turns=1, ele_start=TCCS_name, ele_stop=ele2)
line1.track(part_afterTCCS_21, num_turns=1, ele_start=TCCS_name, ele_stop=ele2)
part_TARGET_30 = part_afterTCCS_20.copy()
part_TARGET_31 = part_afterTCCS_21.copy()
line0.track(part_TARGET_30, num_turns=1, ele_start=ele2, ele_stop=TARGET_name)
line1.track(part_TARGET_31, num_turns=1, ele_start=ele2, ele_stop=TARGET_name)
#part40 = part10.copy()
#line0.track(part40, num_turns=1, ele_start='ip6', ele_stop='ip5')

jaw_L_TARGET = line0.elements[idx_TARGET].jaw_L 
jaw_L_TCCS = line0.elements[idx_TCCS].jaw_L 

ydim_TCCS = coll_dict[TCCS_name]['xdim']
xdim_TCCS =  coll_dict[TCCS_name]['ydim']

       
ydim_TARGET = coll_dict[TARGET_name]['xdim']
xdim_TARGET =  coll_dict[TARGET_name]['ydim']    


"""hit_TCCS_0 = part_TCCS_10.filter((part_TCCS_10.y > line0.elements[idx_TCCS].jaw_L ) & (part_TCCS_10.y < line0.elements[idx_TCCS].jaw_L + ydim_TCCS) & (part_TCCS_10.x > -xdim_TCCS/2) & (part_TCCS_10.x < xdim_TCCS/2) & (part_TCCS_10.state == 1) )
hit_TCCS_1 = part_TCCS_11.filter((part_TCCS_11.y > line1.elements[idx_TCCS].jaw_L) & (part_TCCS_11.y < line1.elements[idx_TCCS].jaw_L + ydim_TCCS) & (part_TCCS_11.x > -xdim_TCCS/2) & (part_TCCS_11.x < xdim_TCCS/2) & (part_TCCS_11.state == 1) )
print(len(hit_TCCS_0.y))
print(len(hit_TCCS_1.y))


hit_TARGET_0 = part_TARGET_30.filter((part_TARGET_30.y > line0.elements[idx_TARGET].jaw_L ) & (part_TARGET_30.y < line0.elements[idx_TARGET].jaw_L + ydim_TARGET) & (part_TARGET_30.x > -xdim_TARGET/2) & (part_TARGET_30.x < xdim_TARGET/2) & (part_TARGET_30.state == 1) )
hit_TARGET_1 = part_TARGET_31.filter((part_TARGET_31.y > line1.elements[idx_TARGET].jaw_L) & (part_TARGET_31.y < line1.elements[idx_TARGET].jaw_L + ydim_TARGET) & (part_TARGET_31.x > -xdim_TARGET/2) & (part_TARGET_31.x < xdim_TARGET/2) & (part_TARGET_31.state == 1) )
print(len(hit_TARGET_0.y))
print(len(hit_TARGET_1.y))"""



hit_TCCS_0 = part_TCCS_10.filter((part_TCCS_10.y > (line0.elements[idx_TCCS].jaw_L + line0.elements[idx_TCCS].ref_y)) & (part_TCCS_10.y < (line0.elements[idx_TCCS].jaw_L + line0.elements[idx_TCCS].ref_y + ydim_TCCS)) & (part_TCCS_10.x > -xdim_TCCS/2) & (part_TCCS_10.x < xdim_TCCS/2) & (part_TCCS_10.state == 1))
hit_TCCS_1 = part_TCCS_11.filter((part_TCCS_11.y > (line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y)) & (part_TCCS_11.y < (line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y + ydim_TCCS)) & (part_TCCS_11.x > -xdim_TCCS/2) & (part_TCCS_11.x < xdim_TCCS/2) & (part_TCCS_11.state == 1))
print(len(hit_TCCS_0.y))
print(len(hit_TCCS_1.y))


hit_TARGET_0 = part_TARGET_30.filter((part_TARGET_30.y > (line0.elements[idx_TARGET].jaw_L + line0.elements[idx_TARGET].ref_y)) & (part_TARGET_30.y < (line0.elements[idx_TARGET].jaw_L + line0.elements[idx_TARGET].ref_y + ydim_TARGET)) & (part_TARGET_30.x > -xdim_TARGET/2) & (part_TARGET_30.x < xdim_TARGET/2) & (part_TARGET_30.state == 1) )
hit_TARGET_1 = part_TARGET_31.filter((part_TARGET_31.y > (line1.elements[idx_TARGET].jaw_L + line1.elements[idx_TARGET].ref_y)) & (part_TARGET_31.y < (line1.elements[idx_TARGET].jaw_L + line1.elements[idx_TARGET].ref_y + ydim_TARGET)) & (part_TARGET_31.x > -xdim_TARGET/2) & (part_TARGET_31.x < xdim_TARGET/2) & (part_TARGET_31.state == 1) )
print(len(hit_TARGET_0.y))
print(len(hit_TARGET_1.y))






jx0,jy0 = calcAction(part_before_TCP,tw0,'tcp.d6r7.b2')
jx1,jy1 = calcAction(part_ip6,tw0,'ip6')
jx2,jy2 = calcAction(part_TCCS_10,tw0,TCCS_name)
jx3,jy3 = calcAction(part_afterTCCS_20,tw0,ele2)
jx4,jy4 = calcAction(part_TARGET_30,tw0,TARGET_name)

print(np.mean(jy0))
print(np.mean(jy1[:-np.sum(part_ip6.state<0)]))
print(np.mean(jy2[:-np.sum(part_TCCS_10.state<0)]))
print(np.mean(jy3[:-np.sum(part_afterTCCS_20.state<0)]))
print(np.mean(jy4[:-np.sum(part_TARGET_30.state<0)]))




TARGET_monitor0 = line0.element_dict['TARGET_monitor']
TARGET_monitor_dict0 = TARGET_monitor0.to_dict()

TCCS_monitor0 = line0.element_dict['TCCS_monitor']
TCCS_monitor_dict0 = TCCS_monitor0.to_dict()
        
"""TARGET_imp_0 = get_df_to_save(TARGET_monitor_dict0, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line0.elements[idx_TARGET].jaw_L ,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_0 = get_df_to_save(TCCS_monitor_dict0, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line0.elements[idx_TCCS].jaw_L , 
                epsilon = 0, num_particles=num_particles, num_turns=num_turns)"""

TARGET_imp_0 = get_df_to_save(TARGET_monitor_dict0, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line0.elements[idx_TARGET].jaw_L + line0.elements[idx_TARGET].ref_y,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_0 = get_df_to_save(TCCS_monitor_dict0, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line0.elements[idx_TCCS].jaw_L + line0.elements[idx_TCCS].ref_y, 
                epsilon = 0, num_particles=num_particles, num_turns=num_turns)




TARGET_monitor1 = line1.element_dict['TARGET_monitor']
TARGET_monitor_dict1 = TARGET_monitor1.to_dict()

TCCS_monitor1 = line1.element_dict['TCCS_monitor']
TCCS_monitor_dict1 = TCCS_monitor1.to_dict()
        
TARGET_imp_1 = get_df_to_save(TARGET_monitor_dict1, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line1.elements[idx_TARGET].jaw_L + line1.elements[idx_TARGET].ref_y,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_1 = get_df_to_save(TCCS_monitor_dict1, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y, 
                epsilon = 0, num_particles=num_particles, num_turns=1)

"""TARGET_imp_1 = get_df_to_save(TARGET_monitor_dict1, x_dim = xdim_TARGET, y_dim = ydim_TARGET, jaw_L = line1.elements[idx_TARGET].jaw_L ,
        epsilon = 0, num_particles=num_particles, num_turns=1)


TCCS_imp_1 = get_df_to_save(TCCS_monitor_dict1, x_dim = xdim_TCCS, y_dim = ydim_TCCS, jaw_L = line1.elements[idx_TCCS].jaw_L , 
                epsilon = 0, num_particles=num_particles, num_turns=1)"""

print(len(hit_TCCS_0.y) == len(TCCS_imp_0.y))
print(len(hit_TCCS_1.y) == len(TCCS_imp_1.y))
print(len(hit_TARGET_0.y) == len(TARGET_imp_0.y))
print(len(hit_TARGET_1.y) == len(TARGET_imp_1.y))

print('LINE TUNE: \t TCCS: ', len(TCCS_imp_0.y), "\t TARGET: ", len(TARGET_imp_0.y))
plotHist(hit_TCCS_0.y, hit_TARGET_0.y, line = [line0.elements[idx_TCCS].jaw_L, line0.elements[idx_TCCS].jaw_L + ydim_TCCS, line0.elements[idx_TARGET].jaw_L, line0.elements[idx_TARGET].jaw_L + ydim_TARGET], density=False)

print('LINE PHASE: \t TCCS: ', len(TCCS_imp_1.y), "\t TARGET: ", len(TARGET_imp_1.y))
#plotHist(hit_TCCS_1.y, hit_TARGET_1.y, line = [line1.elements[idx_TCCS].jaw_L, line1.elements[idx_TCCS].jaw_L + ydim_TCCS, line1.elements[idx_TARGET].jaw_L, line1.elements[idx_TARGET].jaw_L + ydim_TARGET], density = False)
plotHist(TCCS_imp_1.y, TARGET_imp_1.y, line = [line1.elements[idx_TCCS].jaw_L, line1.elements[idx_TCCS].jaw_L + ydim_TCCS, line1.elements[idx_TARGET].jaw_L, line1.elements[idx_TARGET].jaw_L + ydim_TARGET])
plotHist(hit_TCCS_1.y, hit_TCCS_1.y, line = [line0.elements[idx_TCCS].jaw_L, line0.elements[idx_TCCS].jaw_L + ydim_TCCS, line1.elements[idx_TCCS].jaw_L, line1.elements[idx_TCCS].jaw_L + ydim_TCCS])



common_ids0 = np.intersect1d(TCCS_imp_0.particle_id, TARGET_imp_0.parent_particle_id)
print(len(common_ids0), len(TARGET_imp_0.parent_particle_id), len(TCCS_imp_0))

common_ids1 = np.intersect1d(TCCS_imp_1.particle_id, TARGET_imp_1.parent_particle_id)
print(len(common_ids1), len(TARGET_imp_1.parent_particle_id), len(TCCS_imp_1))


target_common_1 = TARGET_imp_1[TARGET_imp_1.particle_id.isin(common_ids1)]
target_not_common_1 = TARGET_imp_1[~TARGET_imp_1.particle_id.isin(common_ids1)]

print(len(target_common_1.y), len(target_not_common_1.y),  len(target_common_1.y)+len(target_not_common_1.y), len(TARGET_imp_1.y))

plotHist(target_common_1.y, target_not_common_1.y, line = [line1.elements[idx_TARGET].jaw_L, line1.elements[idx_TARGET].jaw_L + ydim_TARGET])
plotHist(hit_TARGET_1.y, target_not_common_1.y, line = [line1.elements[idx_TARGET].jaw_L, line1.elements[idx_TARGET].jaw_L + ydim_TARGET])
plotHist(hit_TARGET_1.y, target_common_1.y, line = [line1.elements[idx_TARGET].jaw_L, line1.elements[idx_TARGET].jaw_L + ydim_TARGET])


not_common_at_TCCS = part_afterTCCS_21.filter(np.in1d(part_afterTCCS_21.particle_id,target_not_common_1.particle_id))
print(len(not_common_at_TCCS.y), len(target_not_common_1.y))
plotHist(not_common_at_TCCS.y, part_TCCS_11.y, nbins=1000, line = [line1.elements[idx_TCCS].jaw_L, line1.elements[idx_TCCS].jaw_L + ydim_TCCS], range=(line1.elements[idx_TCCS].jaw_L -0.001, line1.elements[idx_TCCS].jaw_L + ydim_TCCS))


not_common_before_TCCS = part_TCCS_11.filter(np.in1d(part_TCCS_11.particle_id,target_not_common_1.particle_id))
random_id = target_not_common_1.particle_id.iloc[0]
print(random_id)
bad_particle = part_ip6.filter(part_ip6.particle_id == random_id)
line1.track(bad_particle, num_turns=1, ele_start='ip6', ele_stop=TCCS_name)
print(bad_particle.y,not_common_before_TCCS.filter(not_common_before_TCCS.particle_id==random_id).y, line1.elements[idx_TCCS].jaw_L)


lowest_id = TARGET_imp_1.particle_id.iloc[100]
#lowest_id = not_common_before_TCCS.particle_id[np.where(not_common_before_TCCS.y == min(not_common_before_TCCS.y))[0][0]]
bad_particle0 = part_ip6.filter(part_ip6.particle_id == lowest_id)
bad_particle1 = part_ip6.filter(part_ip6.particle_id == lowest_id)
#bad_particle0 = part_ip6.filter(part_ip6.particle_id == random_id)
#bad_particle1 = part_ip6.filter(part_ip6.particle_id == random_id)
s0, s1={}, {}
y0, y1 = {}, {}


idx_ip6 = line1.element_names.index('ip6')
for i in range(idx_ip6, idx_TARGET):
    line1.track(bad_particle1, ele_start=i, ele_stop=i+1)
    s1[line1.element_names[i]] = bad_particle1.s[0]
    y1[line1.element_names[i]] = bad_particle1.y[0]

for i in range(idx_ip6, idx_TARGET):
    line0.track(bad_particle0, ele_start=i, ele_stop=i+1)
    s0[line0.element_names[i]] = bad_particle0.s[0]
    y0[line0.element_names[i]] = bad_particle0.y[0]


#print(line1.elements[idx_TCCS].jaw_L - min(not_common_before_TCCS.y))

import matplotlib.pyplot as plt
plt.plot(s0.values(),y0.values(), color='b')
plt.plot(s1.values(),y1.values(), color='r')
plt.vlines(s1[TCCS_name], 0, 0.03, color='k')
plt.vlines(s1[TCCS_name]- coll_dict[TCCS_name]['length']/2, 0, 0.03, color='k', linestyle='--')
plt.vlines(s1[TARGET_name+'_upstream_aper'], 0, 0.03,color='k')
plt.hlines(line0.elements[idx_TCCS].jaw_L+line0.elements[idx_TCCS].ref_y, s0[TCCS_name], s0[TARGET_name+'_upstream_aper'], color='b', linestyles='--')
#plt.hlines(min(not_common_before_TCCS.y),s1[TCCS_name], s1[TARGET_name+'_upstream_aper'],  color='k', linestyle='--' )
plt.hlines(line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y,s1[TCCS_name], s1[TARGET_name+'_upstream_aper'],  color='r', linestyle='--' )
plt.show()




not_common_after_TCCS = part_afterTCCS_21.filter(np.in1d(part_afterTCCS_21.particle_id,target_not_common_1.particle_id))

plotHist(not_common_before_TCCS.py, not_common_after_TCCS.py)



tccs_targ = np.intersect1d(part_TCCS_11.particle_id, TARGET_imp_1.particle_id)
part_tccs_targ = part_TCCS_11.filter(np.in1d(part_TCCS_11.particle_id, tccs_targ))
part_aftertccs_targ = part_afterTCCS_21.filter(np.in1d(part_afterTCCS_21.particle_id, tccs_targ))
plotHist(part_tccs_targ.py, part_aftertccs_targ.py)




print(line0.elements[idx_TCCS].jaw_L, line0.elements[idx_TCCS].ref_y, line0.elements[idx_TCCS].jaw_L + line0.elements[idx_TCCS].ref_y)
print(line1.elements[idx_TCCS].jaw_L, line1.elements[idx_TCCS].ref_y, line1.elements[idx_TCCS].jaw_L + line1.elements[idx_TCCS].ref_y)


def print_funct(line, tw,  ref_mode = True):
    idx_TCCS = line.element_names.index(TCCS_name)
    idx_TARGET = line.element_names.index(TARGET_name)
    idx_TCCP = line.element_names.index(TCCP_name)
    idx_TCP = line.element_names.index(TCP_name)

    #tw = line.twiss()
    beta_y_TCCS = tw[:,TCCS_name]['bety'][0]
    beta_y_TCCP = tw[:,TCCP_name]['bety'][0]
    beta_y_TARGET = tw[:,TARGET_name]['bety'][0]
    beta_y_TCP = tw[:,TCP_name]['bety'][0]
    beta_rel = line.particle_ref._xobject.beta0[0]
    gamma = line.particle_ref._xobject.gamma0[0]

    emittance_phy = normalized_emittance/(beta_rel*gamma)

    sigma_TCCS = np.sqrt(emittance_phy*beta_y_TCCS)
    sigma_TCCP = np.sqrt(emittance_phy*beta_y_TCCP)
    sigma_TARGET = np.sqrt(emittance_phy*beta_y_TARGET)
    sigma_TCP = np.sqrt(emittance_phy*beta_y_TCP)

   
    if ref_mode:
        print(f"\nTCCS\nCrystalAnalysis(n_sigma={(line.elements[idx_TCCS].jaw_L + line.elements[idx_TCCS].ref_y)/sigma_TCCS}, length={ coll_dict[ TCCS_name]['length']}, ydim={ coll_dict[ TCCS_name]['xdim']}, xdim={ coll_dict[ TCCS_name]['ydim']}, bending_radius={ coll_dict[ TCCS_name]['bending_radius']}, align_angle={ line.elements[idx_TCCS].align_angle}, sigma={sigma_TCCS})")
        print(f"TARGET\nTargetAnalysis(n_sigma={(line.elements[idx_TARGET].jaw_L+ line.elements[idx_TARGET].ref_y)/sigma_TARGET}, length={ coll_dict[ TARGET_name]['length']}, ydim={ coll_dict[ TARGET_name]['xdim']}, xdim={ coll_dict[ TARGET_name]['ydim']}, sigma={sigma_TARGET})")
        print(f"TCCP\nCrystalAnalysis(n_sigma={(line.elements[idx_TCCP].jaw_L+ line.elements[idx_TCCP].ref_y)/sigma_TCCP}, length={ coll_dict[ TCCP_name]['length']}, ydim={ coll_dict[ TCCP_name]['xdim']}, xdim={ coll_dict[ TCCP_name]['ydim']}, bending_radius={ coll_dict[ TCCP_name]['bending_radius']}, align_angle={line.elements[idx_TCCP].align_angle}, sigma={sigma_TCCP})")
        print(f"TCP\nTargetAnalysis(n_sigma={(line.elements[idx_TCP].jaw_L+ line.elements[idx_TCP].ref_y)/sigma_TCP}, length={coll_dict[ TCP_name]['length']}, ydim={0.025}, xdim={0.025}, sigma={sigma_TCP})")
    else:
        print(f"\nTCCS\nCrystalAnalysis(n_sigma={(line.elements[idx_TCCS].jaw_L)/sigma_TCCS}, length={ coll_dict[ TCCS_name]['length']}, ydim={ coll_dict[ TCCS_name]['xdim']}, xdim={ coll_dict[ TCCS_name]['ydim']}, bending_radius={ coll_dict[ TCCS_name]['bending_radius']}, align_angle={ line.elements[idx_TCCS].align_angle}, sigma={sigma_TCCS})")
        print(f"TARGET\nTargetAnalysis(n_sigma={line.elements[idx_TARGET].jaw_L/sigma_TARGET}, length={ coll_dict[ TARGET_name]['length']}, ydim={ coll_dict[ TARGET_name]['xdim']}, xdim={ coll_dict[ TARGET_name]['ydim']}, sigma={sigma_TARGET})")
        print(f"TCCP\nCrystalAnalysis(n_sigma={line.elements[idx_TCCP].jaw_L/sigma_TCCP}, length={ coll_dict[ TCCP_name]['length']}, ydim={ coll_dict[ TCCP_name]['xdim']}, xdim={ coll_dict[ TCCP_name]['ydim']}, bending_radius={ coll_dict[ TCCP_name]['bending_radius']}, align_angle={line.elements[idx_TCCP].align_angle}, sigma={sigma_TCCP})")
        print(f"TCP\nTargetAnalysis(n_sigma={line.elements[idx_TCP].jaw_L/sigma_TCP}, length={coll_dict[ TCP_name]['length']}, ydim={0.025}, xdim={0.025}, sigma={sigma_TCP})")
    