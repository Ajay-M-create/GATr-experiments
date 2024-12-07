# GATr Symmetry Experiment with 20 Points

# Destination: Bizon
export BASEDIR="/home/ajay/GATr-experiments/tmp/gatr-experiments"
echo "Base Directory: ${BASEDIR}"

python ../scripts/symmetry_experiment.py \
    base_dir="${BASEDIR}" \
    seed=42 \
    model=gatr_symmetry \
    data.data_dir="${BASEDIR}/data/symmetry_20" \
    data.subsample=0.05 \
    training.steps=10000 \
    run_name=gatr_symmetry_20 