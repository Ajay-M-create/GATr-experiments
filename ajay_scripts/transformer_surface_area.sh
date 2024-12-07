#dstorm
#export BASEDIR="/storage_bizon/sabrant_rocket_2tb/ajay/repositories/ajay_code/geometric-algebra-transformer/tmp/gatr-experiments"

#bizon
export BASEDIR="/home/ajay/GATr-experiments/tmp/gatr-experiments"
echo $BASEDIR
python ../scripts/surface_area_experiment.py base_dir="${BASEDIR}" seed=42 model=transformer_surface_area data.subsample=0.05 training.steps=10000 run_name=transformer_surface_area