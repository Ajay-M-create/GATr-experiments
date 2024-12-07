# GATr Symmetry Experiment with 0 Points

# Destination: Bizon
export BASEDIR="/home/ajay/GATr-experiments/tmp/gatr-experiments"
echo "Base Directory: ${BASEDIR}"

python ../scripts/symmetry_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gatr_symmetry \
    data.data_dir="${BASEDIR}/data/symmetry_80" \
    data.subsample=0.01 \
    training.steps=5000 \
    run_name=gatr_symmetry_80