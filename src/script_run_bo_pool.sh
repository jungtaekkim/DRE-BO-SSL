#!/bin/bash

PYTHON_SCRIPT='run_bo_pool.py'
NUM_INIT=5
NUM_ITER=500
SEED=42
MODELS='label_propagation label_spreading'
TARGETS='beale branin bukin6 sixhumpcamel natsbench_cifar10-valid natsbench_cifar100 natsbench_ImageNet16-120 digits_mnist tabularbenchmarks_protein tabularbenchmarks_slice tabularbenchmarks_naval tabularbenchmarks_parkinsons'


for MODEL in $MODELS
do
    for TARGET in $TARGETS
    do
        echo $MODEL $TARGET

        python $PYTHON_SCRIPT --model $MODEL --target $TARGET --seed $SEED --num_init $NUM_INIT --num_iter $NUM_ITER
    done
done
