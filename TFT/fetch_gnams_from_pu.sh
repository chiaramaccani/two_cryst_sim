#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <n>"
    exit 1
fi

# Set the remote machine and base directory
REMOTE_MACHINE="twocryst-pu1"
REMOTE_BASE_DIR="/twocryst/rpo/data"

# Define folders and suffixes
FOLDERS=("gnam_ALFA_1_files" "gnam_ALFA_2_files" "gnam_files")
SUFFIXES=("1" "2" "3")

# Local destination folders
LOCAL_DEST="tmp_gnams"
MERGED_DEST="merged_gnams"

# Create local folders if they don't exist
mkdir -p "$LOCAL_DEST"
mkdir -p "$MERGED_DEST"

# Loop through the folders and copy matching files
COPIED_FILES=()
for i in {0..2}; do
    FOLDER=${FOLDERS[$i]}
    SUFFIX=${SUFFIXES[$i]}
    REMOTE_PATH="$REMOTE_BASE_DIR/$FOLDER/gnam_$1.root"
    LOCAL_FILE="$LOCAL_DEST/gnam_$1_$SUFFIX.root"

    # Check if the file exists on the remote machine
    if ssh "$REMOTE_MACHINE" "[ -f $REMOTE_PATH ]"; then
        scp "$REMOTE_MACHINE:$REMOTE_PATH" "$LOCAL_FILE"
        COPIED_FILES+=("$LOCAL_FILE")
    fi
done

# Merge files if at least one was copied
if [ ${#COPIED_FILES[@]} -gt 0 ]; then
    MERGED_FILE="$MERGED_DEST/TOTAL_gnam_$1.root"
    hadd -f "$MERGED_FILE" "${COPIED_FILES[@]}"
    echo "Merged file saved as $MERGED_FILE."
else
    echo "No files found to merge."
fi
