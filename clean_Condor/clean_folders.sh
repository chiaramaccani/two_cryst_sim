#!/bin/bash
#!/bin/bash

# Check if the user provided a directory path
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/parent_directory"
    exit 1
fi

TARGET_DIR="$1"

# Check if the path exists and is a directory
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a valid directory."
    exit 1
fi

# Loop through all items (files and folders) in the target directory
for item in "$TARGET_DIR"/*; do
    name="$(basename "$item")"

    if [[ "$name" != "Job.0" && "$name" != "input_cache" ]]; then
        echo "Deleting: $item"
        rm -rf "$item"
    else
        echo "Keeping: $item"
    fi
done