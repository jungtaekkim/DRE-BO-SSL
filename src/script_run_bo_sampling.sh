#!/bin/bash

PYTHON_SCRIPT='run_bo_sampling.py'
NUM_INIT=5
NUM_ITER=500
SEED=42
INFO_SAMPLING='gaussian_100'
MODELS='label_propagation label_spreading'
TARGETS='beale branin bukin6 sixhumpcamel'


for MODEL in $MODELS
do
    for TARGET in $TARGETS
    do
        echo $MODEL $TARGET

        python $PYTHON_SCRIPT --model $MODEL --target $TARGET --seed $SEED --num_init $NUM_INIT --num_iter $NUM_ITER --info_sampling $INFO_SAMPLING
    done
done
