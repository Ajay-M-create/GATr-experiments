# Transformer Surface Area Experiment with 10 Points

# Destination: Bizon
export BASEDIR="/home/ajay/GATr-experiments/tmp/gatr-experiments"
echo "Base Directory: ${BASEDIR}"

python ../scripts/surface_area_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=transformer_surface_area \
    data.data_dir="${BASEDIR}/data/surface_area_10" \
    data.subsample=0.01 \
    training.steps=5000 \
    run_name=transformer_surface_area_10 