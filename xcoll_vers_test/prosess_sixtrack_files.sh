#!/bin/bash

BJORN_PATH=/afs/cern.ch/work/b/bjlindst/public/sixtrack/inputFiles/crystalCheck/cry1/Hinput
EOS_PATH=/eos/home-c/cmaccani/xsuite_sim/two_cryst_sim/Condor/xcoll_vers_test
AFS_PATH=/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim/xcoll_vers_test




# Define the offset to subtract
retreive_files=0

# Check if the boolean variable is true
if [ $retreive_files -eq 1 ]; then
    echo "Retreive files from Bjorn folder"
    cd $BJORN_PATH
    for dir in run*/; do
        folder_name=${dir}

        # Strip the 'run' prefix and the trailing '/'
        number="${folder_name#run}"
        number="${number%/}"

        #echo "folder ${dir}, number ${number}"

    if [ -f "${dir}deltapx.dat.gz" ]; then
        if [ -f $EOS_PATH/sixtrack_files/deltapx${number}.dat.gz ]; then
            echo "File already exists, skipping"
            continue
        else
            cp "${dir}deltapx.dat.gz" $EOS_PATH/sixtrack_files/deltapx${number}.dat.gz
        fi
    fi
done

else
    echo "Files already retreived, just process data"
fi

cd $EOS_PATH/

unzip_data=0

if [ $unzip_data -eq 1 ]; then
    echo "Unzip data"
    cd sixtrack_files
    for file in deltapx*.dat.gz; do
        if [ -f ${file%.gz} ]; then
            echo "File already unzipped, skipping"
            continue
        else
            gunzip -k $file
            rm $file
        fi
    done
else
    echo "Data already unzipped"
fi



process_data=0

if [ $process_data -eq 1 ]; then

    cd $EOS_PATH/sixtrack_files
    rm ang_*.txt

    for file in *.dat; do
        echo "Processing $file"

        awk 'FNR > 1 && FNR <= 1000001 {
            value = $2 - 5.21380453E-05
            kick = $1
            result = value * 1000000
            intpart = int(result)
            file_name="ang_" intpart "_.txt"
            if (intpart > -50) {
                print value "\t" kick >> file_name
            }
            ii++
            if (ii == 100000) {
                print "Processed 100000 lines, currently on line " NR > "/dev/stderr"
                ii = 0
            }
        }' "$file"

    done
fi
