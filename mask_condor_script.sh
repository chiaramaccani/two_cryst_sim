#!/bin/bash
export HOME_TWOCRYST=/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim
echo $HOME_TWOCRYST
PYTHON_SCRIPT_DIR=/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim

echo /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc11-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc11-opt/setup.sh
echo /afs/cern.ch/user/c/cmaccani/xsuite/venv_xsuite/bin/activate
source /afs/cern.ch/user/c/cmaccani/xsuite/venv_xsuite/bin/activate

CONFIGFILE=%(config_file)s
SEED=%(seed)s
INPUT_ARCHIVE=%(input_cache_archive)s

cp $INPUT_ARCHIVE .
tar -xzvf *.tar.gz

sed -i "s/^  seed:.*/  seed: ${SEED}/g" ${CONFIGFILE}


# print the OS version
if lspci | grep -i "NVIDIA Corporation" &> /dev/null; then
   echo "NVIDIA GPU detected."
   # If NVIDIA GPU is present, load the CUDA-flavored CVMFS release
   if [[ -f /etc/os-release ]]; then
      source /etc/os-release
      if [[ "$VERSION_ID" =~ ^7 ]]; then
            echo "This is CENTOS7"
      elif [[ "$VERSION_ID" =~ ^8 ]]; then
            echo "This is CENTOS8"
      elif [[ "$VERSION_ID" =~ ^9 ]]; then
            echo "This is CENTOS9"
      else
            echo "Unsupported OS: $ID $VERSION_ID"
      fi
   else
      echo "Unable to detect the operating system."
   fi
else
   echo "No NVIDIA GPU detected."
   # If no NVIDIA GPU is detected, load the non-CUDA CVMFS release
   if [[ -f /etc/os-release ]]; then
      source /etc/os-release
      if [[ "$VERSION_ID" =~ ^7 ]]; then
            echo "This is CENTOS7"
      elif [[ "$VERSION_ID" =~ ^8 ]]; then
            echo "This is CENTOS8"
      elif [[ "$VERSION_ID" =~ ^9 ]]; then
            echo "This is CENTOS9"
      else
            echo "Unsupported OS: $ID $VERSION_ID"
      fi
   else
      echo "Unable to detect the operating system."
   fi
fi

echo $(which python)
echo $(which python3)

# you can execute now any command you want. here it's a python script, but it could be anything

#python3 $HOME_TWOCRYST/lossmap_LHC.py  ${CONFIGFILE}  
#python3 $HOME_TWOCRYST/lossmap_LHC_new_version.py  ${CONFIGFILE}  
#python3 $HOME_TWOCRYST/loss_map_bkp.py  ${CONFIGFILE}  
python3 $HOME_TWOCRYST/GR_sim/lossmap_LHC_GR.py  ${CONFIGFILE}  


