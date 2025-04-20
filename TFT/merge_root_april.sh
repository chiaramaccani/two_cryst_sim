#!/bin/bash

# Check if the run number argument is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 <run_number>"
    exit 1
fi

# Get the run number from input
number="$1"

# Set remote and local details
REMOTE_MACHINE="twocryst-pu1"
REMOTE_BASE_DIR="/home/twocryst/oh_snapshots/current"
LOCAL_DEST="tmp_gnams"
MERGED_DEST="merged_gnams"

# Create local folders if they don't exist
mkdir -p "$LOCAL_DEST"
mkdir -p "$MERGED_DEST"

# Get the list of matching remote files
REMOTE_FILES=$(ssh "$REMOTE_MACHINE" "ls ${REMOTE_BASE_DIR}/r${number}"*.root 2>/dev/null)

# Array to store local file paths
COPIED_FILES=()

# Copy each file
i=1
for remote_file in $REMOTE_FILES; do
    filename=$(basename "$remote_file")
    local_file="$LOCAL_DEST/${filename%.root}_$i.root"
    
    echo "Copying $remote_file to $local_file"
    scp "$REMOTE_MACHINE:$remote_file" "$local_file"
    
    COPIED_FILES+=("$local_file")
    ((i++))
done

# Merge files if any were copied
if [ ${#COPIED_FILES[@]} -gt 0 ]; then
    MERGED_FILE="$MERGED_DEST/TOTAL_gnam_$number.root"
    hadd -f "$MERGED_FILE" "${COPIED_FILES[@]}"
    echo "Merged file saved as $MERGED_FILE."
else
    echo "No files found to copy or merge for run number $number."
fi
