#!/bin/bash

# Run experiments in parallel across GPUs
echo "Running all experiments in parallel across GPUs..."

# Create a list of all experiment scripts
cat << 'EOF' > experiment_list.txt
./gatr_volume.sh 0
./gatr_volume_10.sh 0
./gatr_volume_20.sh 1
./gatr_surface_area.sh 0
./gatr_surface_area_10.sh 0
./gatr_surface_area_20.sh 1
./gatr_symmetry.sh 0
./gatr_symmetry_10.sh 0
./gatr_symmetry_20.sh 1
./transformer_volume.sh 1
./transformer_volume_10.sh 1
./transformer_volume_20.sh 1
./transformer_surface_area.sh 1
./transformer_surface_area_10.sh 1
./transformer_surface_area_20.sh 1
./transformer_symmetry.sh 1
./transformer_symmetry_10.sh 1
./transformer_symmetry_20.sh 1
EOF

# Run experiments in parallel using GNU Parallel
parallel --jobs 2 --colsep ' ' 'CUDA_VISIBLE_DEVICES={2} {1}' :::: experiment_list.txt

echo "All experiments completed!"
