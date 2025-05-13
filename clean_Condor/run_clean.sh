#!/bin/bash

# Check if the user provided a top-level directory
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/top_level_directory"
    exit 1
fi

TOP_LEVEL_DIR="$1"

# Check if the top-level directory is valid
if [ ! -d "$TOP_LEVEL_DIR" ]; then
    echo "Error: '$TOP_LEVEL_DIR' is not a valid directory."
    exit 1
fi

# Loop through each subdirectory in the top-level directory
for dir in "$TOP_LEVEL_DIR"/*/; do
    # Remove trailing slash for consistency
    dir="${dir%/}"

    echo "Running clean_folders.sh on $dir"
    . clean_folders.sh "$dir"
done