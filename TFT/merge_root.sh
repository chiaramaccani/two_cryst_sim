#!/bin/bash

# Check if the number argument is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 <run_number>"
    exit 1
fi

# Get the run number from input
number="$1"

# Define base path
base_path="/eos/project-t/twocryst/data/tft/gnamRootFiles/"

# Define folder list
folder_list=("gnam_1TFT" "gnam_2TFT" "gnam_5TFT" "gnamTFT")

# Define output file
output_file="./merged_gnams/TOTAL_gnam_${number}.root"

# Initialize run list
run_list=("hadd" "-f" "$output_file")

# Loop through folders
for folder in "${folder_list[@]}"; do
    path="${base_path}${folder}/gnam_${number}.root"

    if [[ -f "$path" ]]; then
        run_list+=("$path")
    else
        echo "File $path not found."
    fi
done

# Execute the hadd command if files were found
if [[ ${#run_list[@]} -gt 3 ]]; then  # At least one valid file + "hadd -f output_file"
    echo "Merging files..."
    "${run_list[@]}"
else
    echo "No valid input files found. Merging skipped."
fi
