import sys
import os
from pathlib import Path
import yaml
from pylhc_submitter.job_submitter import main as htcondor_submit
import numpy as np
import time
import copy
import gzip

from IPython import embed




import shutil

def resolve_and_cache_paths(iterable_obj, resolved_iterable_obj, cache_destination):
    if isinstance(iterable_obj, (dict, list)):
        for k, v in (iterable_obj.items() if isinstance(iterable_obj, dict) else enumerate(iterable_obj)):
            possible_path = Path(str(v))
            if not isinstance(v, (dict, list)) and possible_path.exists() and possible_path.is_file():
                shutil.copy(possible_path, cache_destination)
                resolved_iterable_obj[k] = possible_path.name
            resolve_and_cache_paths(v, resolved_iterable_obj[k], cache_destination)


def dump_dict_to_yaml(dict_obj, file_path):
        with open(file_path, 'w') as yaml_file:
            yaml.dump(dict_obj, yaml_file, 
                      default_flow_style=False, sort_keys=False)
            


def submit_jobs(config_dict, config_file):
    
    sub_dict = config_dict['jobsubmission']
    workdir = Path(os.path.expandvars(sub_dict['working_directory'])).resolve()
    num_jobs = sub_dict['num_jobs']
    replace_dict_in = sub_dict.get('replace_dict', {})
    executable = sub_dict.get('executable', 'bash')
    mask_abspath = Path(os.path.expandvars(sub_dict['mask'])).resolve()


    workdir = workdir.parent / Path(f"{workdir.name}_{time.strftime('%Y%m%d-%H%M')}")

    input_cache = Path(workdir, 'input_cache')
    os.makedirs(workdir)
    os.makedirs(input_cache)


    print(workdir)

    shutil.copy(config_file, input_cache)

    exclude_keys = {'jobsubmission',} # The submission block is not needed for running
    # Preserve the key order
    reduced_config_dict = {k: config_dict[k] for k in 
                        config_dict.keys() if k not in exclude_keys}
    resolved_config_dict = copy.deepcopy(reduced_config_dict)
    resolve_and_cache_paths(reduced_config_dict['input_files'], resolved_config_dict['input_files'], input_cache)

    conf_fname = Path(config_file).resolve().name

    resolved_conf_file = f'for_jobs_{conf_fname}' # config file used to run each job
    dump_dict_to_yaml(resolved_config_dict, Path(input_cache, resolved_conf_file))

    shutil.make_archive(input_cache, 'gztar', input_cache)

    jobflavour = sub_dict['jobflavour']

    seeds = np.arange(num_jobs) + 1 # Start the seeds at 1
    replace_dict = {'seed': seeds.tolist(), 'config_file': resolved_conf_file, 'input_cache_archive': str(input_cache) + '.tar.gz'}

    processed_opts = {'working_directory', 'num_jobs', 'executable', 'mask', 'jobflavour'}
    submitter_opts = list(set(sub_dict.keys()) - processed_opts)
    submitter_options_dict = { op: sub_dict[op] for op in submitter_opts }
    

    htcondor_submit(
        mask=mask_abspath,
        working_directory=workdir,
        output_destination = 'root://eosuser.cern.ch//eos/user/c/cmaccani/xsuite_sim/two_cryst_sim/Condor/' + Path(workdir).name,
        executable=executable,
        replace_dict=replace_dict,
        jobflavour = jobflavour,
        **submitter_options_dict)

def main():

    config_file = sys.argv[1]
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    sub_dict = config_dict['jobsubmission']
    keys_to_drop = [  'angular_scan', 'angular_scan_range_lower', 'angular_scan_range_upper', 'angular_scan_step' ]
    angular_scan = sub_dict['angular_scan'] if sub_dict['angular_scan'] != 'None' else None
    if angular_scan is not None:
        lower_bound = sub_dict['angular_scan_range_lower']
        upper_bound = sub_dict['angular_scan_range_upper']
        step = sub_dict['angular_scan_step']
        angles_list = np.arange(lower_bound, upper_bound, step)
        config_dict_to_submit = copy.deepcopy(config_dict)
        for angle in angles_list:
            if sub_dict['angular_scan'] == 'TCCS':
                config_dict_to_submit['run']['TCCS_align_angle_step'] = float(angle)
            elif sub_dict['angular_scan'] == 'TCCP':    
                config_dict_to_submit['run']['TCCP_align_angle_step'] = float(angle)
            elif sub_dict['angular_scan'] == 'CRY':
                config_dict_to_submit['run']['CRY_align_angle_step'] = float(angle)
            else:
                raise ValueError('angular_scan must be TCCS, TCCP or CRY')
            config_dict_to_submit['jobsubmission']['working_directory'] = sub_dict['working_directory'] + f'_{angle*1e6:.{1}f}_'
            for key in keys_to_drop:
                if key in config_dict_to_submit['jobsubmission']:
                    del config_dict_to_submit['jobsubmission'][key]
            submit_jobs(config_dict_to_submit, config_file)  
    else:
        for key in keys_to_drop:
                if key in config_dict['jobsubmission']:
                    del config_dict['jobsubmission'][key]
        submit_jobs(config_dict, config_file)

if __name__ == '__main__':

    main()