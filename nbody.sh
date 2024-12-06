export BASEDIR="/storage_bizon/sabrant_rocket_2tb/ajay/repositories/ajay_code/geometric-algebra-transformer/tmp/gatr-experiments"
echo $BASEDIR
python scripts/nbody_experiment.py base_dir="${BASEDIR}" seed=42 model=gatr_nbody data.subsample=0.01 training.steps=5000 run_name=gatr