import sys
from pathlib import Path
import yaml
import numpy as np
import os
import lossmaps as lm
import subprocess 
import matplotlib.pyplot as plt
import pandas as pd



#   ----------------------------------   SUBMIT SCAN     --------------------------------------------------------------------------

def submit_scan(config_file, lower_limit = -10e-6, upper_limit = 10e-6, step = 1e-6):

    #config_file = sys.argv[1]
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)         
    
    scan_range = np.arange(lower_limit, upper_limit, step)

    name_splitted = config_file.split('.yaml')
    working_directory = config_dict['jobsubmission']['working_directory'] #'/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim/Condor/TEST_B2V_align_test_CRY6.5_'
    job_submitter_script = '/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim/Submit_HTCondor.py'

    for _angle in scan_range:

        angle = round(_angle, 6)
        config_dict['run']['TTCS_align_angle_step'] = float(angle)

        config_dict['jobsubmission']['working_directory'] = working_directory + str(int(round(_angle*1e6))) + '_'

        name_changed_file = 'input_files/angular_scan/'+ name_splitted[0] + str(int(round(_angle*1e6)))+'.yaml'
        old_location = config_dict['input_files']['collimators']
        new_config_file = old_location.split('input')[0] + name_changed_file

        with open(new_config_file, 'w') as stream:
            try:
                yaml.dump(config_dict, stream, default_flow_style=False)
            except yaml.YAMLError as exc:
                print(exc)

        print(new_config_file)
        os.system("python " + job_submitter_script + ' ' + new_config_file)



#   ----------------------------------   ANALYSIS     --------------------------------------------------------------------------



def analysis_scan(prefix_name = 'TEST_B2V_align_test_CRY5.0', config_file = 'config_sim.yaml'):

    Condor_path = '/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim/Condor/'
    test_list = [Condor_path + i for i in os.listdir(Condor_path) if prefix_name in i]
    
    theta = []
    losses_at_CRY1 = []
    losses_at_TCP = []
    losses_relative = []
    losses_at_target = []
    losses_at_absorber = []

    TCP_name = 'tcp.d6r7.b2'
    TCCS_name = 'tccs.5r3.b2'
    Target_name = 'target.4l3.b2'
    TCLA_name = 'tcla.a5l3.b2'

    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    coll_file = config_dict['input_files']['collimators']
    with open(coll_file, 'r') as stream:
        coll_dict = yaml.safe_load(stream)['collimators']['b'+config_dict['run']['beam']]


    TCP_length = coll_dict[TCP_name]['length']
    TCCS_length = coll_dict[TCCS_name]['length']
    Target_length = coll_dict[Target_name]['length']
    TCLA_length = coll_dict[TCLA_name]['length']


    for test_name in test_list:

        job_directories = [test_name + '/' + i for i in os.listdir(test_name) if 'Job.' in i]
        n_jobs = len(job_directories)
        losses_json = []
        for job_dir in job_directories:
            if os.path.exists(job_dir +'/Outputdata'):
                loss_file = [filename for filename in os.listdir(job_dir +'/Outputdata') if filename.startswith("lossmap")]
                if loss_file:
                    losses_json.append(job_dir +'/Outputdata/' + loss_file[0])

        n_jobs_verify  = len(losses_json)
        if n_jobs != n_jobs_verify:
            print("!!! Succesful Jobs: ", n_jobs_verify, '/', n_jobs, ' in file: ', test_name)

        if losses_json:
            tmp_name = test_name.split('/Condor/')[1] 
            outfile_name = 'Lossmap_' + tmp_name.split('__')[0]
            angle = float(outfile_name.split('test_')[1].split('_')[1])
            theta.append(angle)

            ThisLM = lm.SimulatedLossMap(lmtype=lm.LMType.B2V, machine=lm.Machine.LHC)
            ThisLM.load_data_json(json_files=losses_json)

            lm.plot_lossmap(ThisLM, zoom=True, outfile = "./Outputdata/Lossmap_outputs/"+outfile_name+'.png')

            losses_df = ThisLM._losses
            #print(losses_df[losses_df['name']=='tccs.5r3.b2']['losses'], '\t', angle )     #tccs.5r3.b2
            loss_cry1 = int(losses_df[losses_df['name']== TCCS_name]['losses']) * TCCS_length                                        #tccs.5r3.b2
            #print(loss_cry1, tmp_name)
            loss_tcp = int(losses_df[losses_df['name']== TCP_name ]['losses']) * TCP_length
            print('angle: ', angle)
            print(losses_df[losses_df['name']== TCLA_name]['losses'])
            #
            loss_at_target = int(losses_df[losses_df['name']== Target_name]['losses'] * Target_length)   

            loss_at_absorber = losses_df[losses_df['name'] == TCLA_name]['losses']
            if len(loss_at_absorber)>0:
                losses_at_absorber.append(int(loss_at_absorber)*TCLA_length)
                print('TCLA LOSSES : ', int(losses_df[losses_df['name']==TCLA_name]['losses'])*TCLA_length, '\n')
            else:
                 losses_at_absorber.append(0)
            losses_at_CRY1.append(loss_cry1)
            losses_at_TCP.append(loss_tcp)
            losses_relative.append(loss_cry1/loss_tcp)
            losses_at_target.append(loss_at_target)
            plt.close()


    print('theta: ', len(theta), '\t\t', theta)
    print('losses_at_target: ', len(losses_at_target), '\t\t', losses_at_target)

    scan_df = pd.DataFrame(np.column_stack([theta, losses_at_CRY1, losses_at_TCP, losses_relative, losses_at_target, losses_at_absorber]),
                                columns =['theta', 'losses_at_CRY1', 'losses_at_TCP', 'losses_relative', 'losses_at_target', 'losses_at_absorber'])

    scan_df.sort_values(by='theta', inplace=True)

    print(scan_df)

    plt.close()

    n_sig = prefix_name.split('CRY')[1]

    fig1 = plt.figure( figsize=(24, 5))
    ax1 = fig1.add_subplot(1,3,1)
    ax1.plot(scan_df['theta'], scan_df['losses_at_CRY1'])
    ax1.set_xlabel('Rotation angle wrt ' + n_sig + r'$\sigma$ envelope [$\mu$rad]')
    ax1.set_ylabel("Losses at CRY1")

    ax2 = fig1.add_subplot(1,3,2)
    ax2.plot(scan_df['theta'], scan_df['losses_relative']) 
    ax2.set_xlabel('Rotation angle wrt ' + n_sig + r'$\sigma$ envelope [$\mu$rad]')
    ax2.set_ylabel('Losses at CRY1 / Losses at TCP')


    ax3 = fig1.add_subplot(1,3,3)
    ax3.plot(scan_df['theta'], scan_df['losses_at_absorber']) 
    ax3.set_xlabel('Rotation angle wrt ' + n_sig + r'$\sigma$ envelope [$\mu$rad]')
    ax3.set_ylabel('Losses at TCLA')
    

    fig1.savefig("./Outputdata/Lossmap_outputs/Angular_scan_"+ str(n_sig) + ".png")
    # inital angle: 1.9235399773503744e-05



    fig2 = plt.figure( figsize=(10, 8))
    ax = fig2.add_subplot(1,1,1)
    ax.plot(scan_df['theta'], scan_df['losses_at_CRY1'], label = 'TCCS')
    ax.plot(scan_df['theta'], scan_df['losses_at_absorber'], label ='TCLA') 
    ax.set_xlabel('Rotation angle wrt ' + n_sig + r'$\sigma$ envelope [$\mu$rad]')
    ax.set_ylabel("Losses")
    ax.legend(loc='center right')
    fig2.savefig("./Outputdata/Lossmap_outputs/Angular_scan_"+ str(n_sig) + "_sovrapp.png")

    return scan_df


#   ----------------------------------   PROTON ON TARGET     --------------------------------------------------------------------------



def proton_on_target(analysis = True, sig_lower = 5.0, sig_upper = 6.5, prefix_name = 'TEST_B2V_align_test_CRY', config_file = 'config_sim.yaml'):

    sigma_list = np.arange(sig_lower, sig_upper + 0.5, 0.5)
    sigmas = [np.round(i, 1) for i in sigma_list]
    pot_filename = "./Outputdata/Lossmap_outputs/proton-on-target.txt"
    print(sigmas)

    if analysis:
        for s in sigmas:
            name = prefix_name+str(s) 
            print(name)
            pot = analysis_scan(name)
            with open(pot_filename, "a") as myfile:
                for index, row in pot.iterrows():
                    line = str(s) + '\t\t' + str(row['theta']) + '\t\t' +  str(row['losses_at_target']) + '\t\t' + str(row['losses_at_CRY1']) + '\t\t' + str(row['losses_relative']) +'\n'
                    print(line)
                    myfile.write(line)


    cols = ['sigma','theta','loss_target','loss_cry', 'loss_relative']
    df = pd.read_csv('./Outputdata/Lossmap_outputs/proton-on-target.txt', header=None, names=cols, sep='\t\t')
    min_cry_losses = df.loc[df.groupby('sigma').loss_cry.idxmin()]
    
    fig1 = plt.figure( figsize=(10, 5))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_yscale('log')
    ax1.set_xticks(sigma_list)
    ax1.plot(min_cry_losses['sigma'], min_cry_losses['loss_target'])
    ax1.set_xlabel(r'CRY1 position [$\sigma$]')
    ax1.set_ylabel("Protons on target (CRY in channeling)")


    df5 = df[df['sigma']==5.0]
    fig2 = plt.figure( figsize=(10, 5))
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(df5['theta'], df5['loss_target']) 
    ax2.set_xlabel(r'Rotation angle wrt 5$\sigma$ envelope [$\mu$rad]')
    ax2.set_ylabel(r'Protons on target with CRY1 at 5$\sigma$ ')

    

    fig1.savefig("./Outputdata/Lossmap_outputs/pot.png")
    fig2.savefig("./Outputdata/Lossmap_outputs/pot_5sig.png")












#   ----------------------------------   MAIN     --------------------------------------------------------------------------

def main():

    if sys.argv[1] == '--submit':
        config_file = sys.argv[2] 
        lower_limit = float(sys.argv[3]) if len(sys.argv) > 3 else -10e-6
        upper_limit = float(sys.argv[4]) if len(sys.argv) > 4 else 10e-6
        step = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6

        submit_scan(config_file, lower_limit, upper_limit, step)

    elif sys.argv[1] == '--analysis':
        prefix_name = sys.argv[2] if len(sys.argv) > 2 else 'TEST_B2V_align_test_CRY5.0_'
        config_file = sys.argv[3] if len(sys.argv) > 3 else 'config_sim.yaml'
        analysis_scan(prefix_name, config_file)
    
    elif sys.argv[1] == '--pot':
        if_analysis = bool(sys.argv[2]) if len(sys.argv) > 2 else False
        lower_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        upper_limit = float(sys.argv[4]) if len(sys.argv) > 4 else 6.5
        prefix_name = sys.argv[5] if len(sys.argv) > 5 else 'TEST_B2V_align_test_CRY'
        config_file = sys.argv[6] if len(sys.argv) > 6 else 'config_sim.yaml'
        proton_on_target(if_analysis, lower_limit, upper_limit, prefix_name, config_file)


    else:
        raise ValueError('The mode must be one of --submit, --analysis')

if __name__ == '__main__':

    main()