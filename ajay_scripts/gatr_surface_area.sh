#dstorm
#export BASEDIR="/storage_bizon/sabrant_rocket_2tb/ajay/repositories/ajay_code/geometric-algebra-transformer/tmp/gatr-experiments"

# GATr Symmetry Experiment with 5 Points

# Destination: Bizon
export BASEDIR="/home/ajay/GATr-experiments/tmp/gatr-experiments"
echo "Base Directory: ${BASEDIR}"

python ../scripts/surface_area_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gatr_surface_area \
    data.data_dir="${BASEDIR}/data/surface_area" \
    data.subsample=0.05 \
    training.steps=10000 \
    run_name=gatr_surface_area 