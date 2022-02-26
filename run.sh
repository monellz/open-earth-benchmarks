#!/bin/bash
#set -x
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd) 
srun -n 1 ${SCRIPT_DIR}/build/fastwavesuv
srun -n 1 ${SCRIPT_DIR}/build/fvtp2d
srun -n 1 ${SCRIPT_DIR}/build/hadvuv
srun -n 1 ${SCRIPT_DIR}/build/hadvuv5th
srun -n 1 ${SCRIPT_DIR}/build/hdiffsa
srun -n 1 ${SCRIPT_DIR}/build/hdiffsmag
srun -n 1 ${SCRIPT_DIR}/build/laplace
srun -n 1 ${SCRIPT_DIR}/build/nh_p_grad
srun -n 1 ${SCRIPT_DIR}/build/p_grad_c
srun -n 1 ${SCRIPT_DIR}/build/uvbke