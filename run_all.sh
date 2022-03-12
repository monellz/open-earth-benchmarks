#!/bin/bash
set -x
ALGO=0
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd) 
srun -n 1 ${SCRIPT_DIR}/build/fastwavesuv ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/fvtp2d ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/hadvuv ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/hadvuv5th ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/hdiffsa ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/hdiffsmag ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/laplace ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/nh_p_grad ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/p_grad_c ${ALGO}
srun -n 1 ${SCRIPT_DIR}/build/uvbke ${ALGO}