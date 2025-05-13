#!/bin/bash

EOS_FOLDER="/eos/home-c/cmaccani/xsuite_sim/two_cryst_sim/Condor"
AFS_FOLDER="/afs/cern.ch/work/c/cmaccani/xsuite_sim/twocryst_sim/Condor"

DRY_RUN=false

# Handle optional --dry-run argument
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Running in DRY-RUN mode: No folders will be deleted."
fi

# Check if directories exist
if [ ! -d "$EOS_FOLDER" ]; then
    echo "Error: '$EOS_FOLDER' is not a valid directory."
    exit 1
fi

if [ ! -d "$AFS_FOLDER" ]; then
    echo "Error: '$AFS_FOLDER' is not a valid directory."
    exit 1
fi

# Count folders
count_eos=$(find "$EOS_FOLDER" -mindepth 1 -maxdepth 1 -type d | wc -l)
count_afs=$(find "$AFS_FOLDER" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Number of folders in '$EOS_FOLDER': $count_eos"
echo "Number of folders in '$AFS_FOLDER': $count_afs"

# Get list of folder names in EOS_FOLDER
eos_folders=()
while IFS= read -r -d '' dir; do
    eos_folders+=("$(basename "$dir")")
done < <(find "$EOS_FOLDER" -mindepth 1 -maxdepth 1 -type d -print0)

# Use associative array for fast lookup
declare -A eos_folder_map
for name in "${eos_folders[@]}"; do
    eos_folder_map["$name"]=1
done

# Loop through AFS folders and delete if not in EOS
deleted=0
while IFS= read -r -d '' afs_dir; do
    afs_name="$(basename "$afs_dir")"
    if [[ -z "${eos_folder_map[$afs_name]}" ]]; then
        if $DRY_RUN; then
            echo "[DRY-RUN] Would delete: $afs_dir"
        else
            echo "Deleting: $afs_dir"
            rm -rf "$afs_dir"
        fi
        ((deleted++))
    else
        echo "Keeping: $afs_dir"
    fi
done < <(find "$AFS_FOLDER" -mindepth 1 -maxdepth 1 -type d -print0)

if $DRY_RUN; then
    echo "DRY-RUN complete. $deleted folder(s) **would be deleted**."
else
    echo "Done. $deleted folder(s) deleted from AFS."
fi
