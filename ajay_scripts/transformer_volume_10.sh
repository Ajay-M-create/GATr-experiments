# Transformer Volume Experiment with 10 Points

# Destination: Bizon
export BASEDIR="/home/ajay/GATr-experiments/tmp/gatr-experiments"
echo "Base Directory: ${BASEDIR}"

python ../scripts/volume_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=transformer_volume \
    data.data_dir="${BASEDIR}/data/volume_10" \
    data.subsample=0.01 \
    training.steps=5000 \
    run_name=transformer_volume_10 