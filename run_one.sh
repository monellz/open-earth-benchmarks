#!/bin/bash
set -x
#export OMP_DISPLAY_ENV=TRUE
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd) 
srun -n 1 ${SCRIPT_DIR}/build/$1 $2 $3 $4