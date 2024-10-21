#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate fcr-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name train_epoch-1000"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --data_path $DATA/path/to/datasets" 
PYARGS="$PYARGS --gpu 0" 

PYARGS="$PYARGS --omega1 1.0"
PYARGS="$PYARGS --omega2 1.0"
PYARGS="$PYARGS --omega3 1.0"


PYARGS="$PYARGS --outcome_dist normal"
PYARGS="$PYARGS --dist_mode match"
PYARGS="$PYARGS --decode_aggr dot"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --batch_size 1024"
PYARGS="$PYARGS --eval_mode classic"

python main.py $PYARGS