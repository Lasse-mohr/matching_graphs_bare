#!/bin/bash

# Create the base directory
mkdir -p data_new_struct

# Create the atlases directory
mkdir -p data_new_struct/atlas_coordinates

# Copy the coordinate files to the atlases directory
cp coordinates_Schaefer2.npy data_new_struct/atlas_coordinates/
cp coordinates_Gla358.npy data_new_struct/atlas_coordinates/

# Loop through each user directory
for user_dir in data/sub-*; do
    user_id=$(basename $user_dir)
    
    # Loop through each file of interest
    for file in $user_dir/connectome/*.npy; do
        # Extract file details using awk
        filename=$(basename $file)
        atlas=$(echo $filename | awk -F'atlas-' '{print $2}' | awk -F'_' '{print $1}')
        type=$(echo $filename | awk -F'_' '{print $2}')
        
        # Skip run-2 files
        if [[ $filename == *"run-2"* ]]; then
            continue
        fi
        
        # Create target directory
        target_dir="data/$atlas/$user_id/$type"
        mkdir -p $target_dir
        
        # Copy the file to the target directory
        cp $file $target_dir/
    done
done

