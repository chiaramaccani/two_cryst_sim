#!/bin/bash

# Usage check
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <run_number> <start_time> <end_time>"
    echo "Example: $0 1750484801 08:02:00 08:04:00"
    exit 1
fi

number="$1"
start_time="$2"
end_time="$3"

# Convert start and end time to seconds since midnight for comparison
start_sec=$(date -d "$start_time" +%s)
end_sec=$(date -d "$end_time" +%s)

REMOTE_MACHINE="twocryst-pu1"
REMOTE_BASE_DIR="/home/twocryst/oh_snapshots/current"
LOCAL_DEST="tmp_gnams"
MERGED_DEST="merged_gnams"

mkdir -p "$LOCAL_DEST"
mkdir -p "$MERGED_DEST"

# Get list of matching remote files
REMOTE_FILES=$(ssh "$REMOTE_MACHINE" "ls ${REMOTE_BASE_DIR}/r${number}"*.root 2>/dev/null)

COPIED_FILES=()
i=1

for remote_file in $REMOTE_FILES; do
    filename=$(basename "$remote_file")
    
    # Extract timestamp from filename: YYYY-MM-DD-HH-MM-SS
    timestamp=$(echo "$filename" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}')
    file_time=$(echo "$timestamp" | awk -F'-' '{print $4":"$5":"$6}')
    
    # Convert file time to seconds since midnight
    file_sec=$(date -d "$file_time" +%s)

    # Check if file time is within the range
    if [[ $file_sec -ge $start_sec && $file_sec -le $end_sec ]]; then
        local_file="$LOCAL_DEST/${filename%.root}_$i.root"
        echo "Copying $remote_file to $local_file"
        scp "$REMOTE_MACHINE:$remote_file" "$local_file"
        COPIED_FILES+=("$local_file")
        ((i++))
    fi
done

if [ ${#COPIED_FILES[@]} -gt 0 ]; then
    # Sanitize time strings for filenames
    start_label=${start_time//:/-}
    end_label=${end_time//:/-}
    MERGED_FILE="$MERGED_DEST/TOTAL_gnam_${number}_${start_label}_${end_label}.root"
    
    hadd -f "$MERGED_FILE" "${COPIED_FILES[@]}"
    echo "Merged file saved as $MERGED_FILE."
else
    echo "No files in the specified time range for run number $number."
fi