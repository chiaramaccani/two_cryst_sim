    # ---------------------------- SAVE DATA ----------------------------
    #print("... Saving impacts on particle data\n")
    df_part = part.to_pandas()
    drop_list = ['chi', 'charge_ratio', 'pdg_id', 'rvv', 'rpp', '_rng_s1', '_rng_s2', '_rng_s3', '_rng_s4', 'weight', 'ptau', 'q0','gamma0','beta0', 'mass0', 'start_tracking_at_element', 's']
    float_variables = ['zeta', 'x', 'px', 'y', 'py', 'delta', 'p0c']
    int_variables = ['at_turn', 'particle_id', 'at_element', 'state', 'parent_particle_id']
    df_part.drop(drop_list, axis=1, inplace=True)
    df_part[float_variables] = df_part[float_variables].astype('float32')
    df_part[int_variables] = df_part[int_variables].astype('int32')
    #df_part.to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='particles', format='table', mode='a',
    #          complevel=9, complib='blosc')



    if 'TCP_generated' in save_list:
        
        print("\n... Saving particles generated in interactions with TCP \n")
        epsilon = 0 #2e10-4
        TCP_monitor_dict = TCP_monitor.to_dict()
        df = pd.DataFrame(TCP_monitor_dict['data'])
        variables = float_variables + int_variables
        variables.remove('at_element')
        var_dict = {}

        for var in variables:
            new_arr = np.array(df[var])
            new_arr = new_arr.reshape((num_particles, num_turns))
            var_dict[var] = new_arr   
        del df
        
        jaw_L_TCP = line.elements[idx_TCP].jaw_L

        
        impact_part_dict = {}
        for key in var_dict.keys():
            impact_part_dict[key] = []

        for part in range(num_particles):
            turn = 0
            if var_dict['state'][part, turn] > 0  and var_dict['y'][part, 0]> (jaw_L_TCP - epsilon):
                for key in var_dict.keys():
                    impact_part_dict[key].append(var_dict[key][part, turn])
        impact_part_df = pd.DataFrame(impact_part_dict) 
        
        impact_part_df.rename(columns={'state': 'this_state'}, inplace=True)
        impact_part_df.rename(columns={'at_turn': 'this_turn'}, inplace=True)
        impact_part_df = pd.merge(impact_part_df, df_part[['at_element', 'state', 'at_turn', 'particle_id']], on='particle_id', how='left')
        
        impact_part_df[float_variables] = impact_part_df[float_variables].astype('float32')
        impact_part_df[int_variables] = impact_part_df[int_variables].astype('int32')
        impact_part_df['this_turn'] = impact_part_df['this_turn'].astype('int32')
        impact_part_df['this_state'] = impact_part_df['this_state'].astype('int32')

        impact_part_df.to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='TCCS_impacts', format='table', mode='a',
                complevel=9, complib='blosc')


    if 'TCCS_impacts' in save_list:
        # SAVE IMPACTS ON TCCS
        print("... Saving impacts on TCCS\n")
        epsilon = 0 #2e10-4
        TCCS_monitor_dict = TCCS_monitor.to_dict()
        df = pd.DataFrame(TCCS_monitor_dict['data'])
        variables = float_variables + int_variables
        variables.remove('at_element')
        var_dict = {}

        for var in variables:
            new_arr = np.array(df[var])
            new_arr = new_arr.reshape((num_particles, num_turns))
            var_dict[var] = new_arr   
        del df
        
        ydim_TCCS = coll_dict[TCCS_name]['xdim']
        xdim_TCCS =  coll_dict[TCCS_name]['ydim']
        jaw_L_TCCS = line.elements[idx_TCCS].jaw_L

        abs_y_low_TCCS = jaw_L_TCCS
        abs_y_up_TCCS = jaw_L_TCCS + ydim_TCCS
        abs_x_low_TCCS = -xdim_TCCS/2
        abs_x_up_TCCS = xdim_TCCS/2
        
        impact_part_dict = {}
        for key in var_dict.keys():
            impact_part_dict[key] = []

        for part in range(num_particles):
            for turn in range(num_turns):
                if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low_TCCS - epsilon) and var_dict['x'][part, turn] < (abs_x_up_TCCS + epsilon) and var_dict['y'][part, turn]> (abs_y_low_TCCS - epsilon) and var_dict['y'][part, turn] < (abs_y_up_TCCS + epsilon):
                    for key in var_dict.keys():
                        impact_part_dict[key].append(var_dict[key][part, turn])
        impact_part_df = pd.DataFrame(impact_part_dict) 
        
        impact_part_df.rename(columns={'state': 'this_state'}, inplace=True)
        impact_part_df.rename(columns={'at_turn': 'this_turn'}, inplace=True)
        impact_part_df = pd.merge(impact_part_df, df_part[['at_element', 'state', 'at_turn', 'particle_id']], on='particle_id', how='left')
        
        impact_part_df[float_variables] = impact_part_df[float_variables].astype('float32')
        impact_part_df[int_variables] = impact_part_df[int_variables].astype('int32')
        impact_part_df['this_turn'] = impact_part_df['this_turn'].astype('int32')
        impact_part_df['this_state'] = impact_part_df['this_state'].astype('int32')

        impact_part_df.to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='TCCS_impacts', format='table', mode='a',
                complevel=9, complib='blosc')



    if 'TARGET_impacts' in save_list:

        # SAVE IMPACTS ON TARGET
        print("... Saving impacts on TARGET\n")

        epsilon = 2.5e-3
        TARGET_monitor_dict = TARGET_monitor.to_dict()
        df = pd.DataFrame(TARGET_monitor_dict['data'])
        var_dict = {}

        for var in variables:
            new_arr = np.array(df[var])
            new_arr = new_arr.reshape((num_particles, num_turns))
            var_dict[var] = new_arr   
        del df
        
        impact_part_dict = {}
        for key in var_dict.keys():
            impact_part_dict[key] = []


        ydim_TARGET = coll_dict[TARGET_name]['xdim']
        xdim_TARGET =  coll_dict[TARGET_name]['ydim']
        jaw_L_TARGET = line.elements[idx_TARGET].jaw_L

        abs_y_low_TARGET = jaw_L_TARGET
        abs_y_up_TARGET = jaw_L_TARGET + ydim_TARGET
        abs_x_low_TARGET = -xdim_TARGET/2
        abs_x_up_TARGET = xdim_TARGET/2

        
        for part in range(num_particles):
            for turn in range(num_turns):
                if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low_TARGET - epsilon) and var_dict['x'][part, turn] < (abs_x_up_TARGET + epsilon) and var_dict['y'][part, turn] > (abs_y_low_TARGET - epsilon) and var_dict['y'][part, turn] < (abs_y_up_TARGET + epsilon):
                    for key in var_dict.keys():
                        impact_part_dict[key].append(var_dict[key][part, turn])
        impact_part_df = pd.DataFrame(impact_part_dict) 
        
        impact_part_df.rename(columns={'state': 'this_state'}, inplace=True)
        impact_part_df.rename(columns={'at_turn': 'this_turn'}, inplace=True)
        impact_part_df = pd.merge(impact_part_df, df_part[['at_element', 'state', 'at_turn', 'particle_id']], on='particle_id', how='left')
        
        impact_part_df[float_variables] = impact_part_df[float_variables].astype('float32')
        impact_part_df[int_variables] = impact_part_df[int_variables].astype('int32')
        impact_part_df['this_turn'] = impact_part_df['this_turn'].astype('int32')
        impact_part_df['this_state'] = impact_part_df['this_state'].astype('int32')

        impact_part_df.to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='TARGET_impacts', format='table', mode='a',
                complevel=9, complib='blosc')


    if 'PIXEL_impacts' in save_list:

        # SAVE IMPACTS ON PIXEL
        print("... Saving impacts on PIXEL\n")

        PIXEL_monitor_dict = PIXEL_monitor.to_dict()
        df = pd.DataFrame(PIXEL_monitor_dict['data'])
        var_dict = {}

        for var in variables:
            new_arr = np.array(df[var])
            new_arr = new_arr.reshape((num_particles, num_turns))
            var_dict[var] = new_arr   
        del df

        
        impact_part_dict = {}
        for key in var_dict.keys():
            impact_part_dict[key] = []

        ydim_PIXEL = coll_dict[TARGET_name]['xdim']
        xdim_PIXEL =  coll_dict[TARGET_name]['ydim']
        jaw_L_PIXEL = sigma_PIXEL * PIXEL_gap

        abs_y_low_PIXEL = jaw_L_PIXEL
        abs_y_up_PIXEL = jaw_L_PIXEL + ydim_PIXEL
        abs_x_low_PIXEL = -xdim_PIXEL/2
        abs_x_up_PIXEL = xdim_PIXEL/2

        epsilon = 3.5e-3
        for part in range(num_particles):
            for turn in range(num_turns):
                if var_dict['state'][part, turn] > 0 and var_dict['x'][part, turn] > (abs_x_low_PIXEL - epsilon) and var_dict['x'][part, turn] < (abs_x_up_PIXEL + epsilon) and var_dict['y'][part, turn] > (abs_y_low_PIXEL - epsilon) and var_dict['y'][part, turn] < (abs_y_up_PIXEL + epsilon):
                    for key in var_dict.keys():
                        impact_part_dict[key].append(var_dict[key][part, turn])
        impact_part_df = pd.DataFrame(impact_part_dict) 
        
        impact_part_df.rename(columns={'state': 'this_state'}, inplace=True)
        impact_part_df.rename(columns={'at_turn': 'this_turn'}, inplace=True)
        impact_part_df = pd.merge(impact_part_df, df_part[['at_element', 'state', 'at_turn', 'particle_id']], on='particle_id', how='left')
        
        impact_part_df[float_variables] = impact_part_df[float_variables].astype('float32')
        impact_part_df[int_variables] = impact_part_df[int_variables].astype('int32')
        impact_part_df['this_turn'] = impact_part_df['this_turn'].astype('int32')
        impact_part_df['this_state'] = impact_part_df['this_state'].astype('int32')

        impact_part_df.to_hdf(Path(path_out,f'particles_B{beam}{plane}.h5'), key='TARGET_impacts', format='table', mode='a',
                complevel=9, complib='blosc')