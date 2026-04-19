#!/bin/bash
#SBATCH --job-name=film_joint_cot
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/job11_film_joint_cot_%j.out
#SBATCH --error=slurm_logs/job11_film_joint_cot_%j.err
#SBATCH --time=12:00:00

# ── Job 11: film, joint fine-tuning, cot_generated ──

set -euo pipefail

cd /home/sgoudarzi/lang_conditioned_neural_ops
mkdir -p slurm_logs

source /home/sgoudarzi/miniconda/etc/profile.d/conda.sh
conda activate base

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "=== JAX devices ==="
BACKEND=$(python -c "import jax; print(jax.default_backend())")
echo "Backend: $BACKEND"
python -c "import jax; print('Devices:', jax.devices())"
if [ "$BACKEND" != "gpu" ]; then
    echo "ERROR: JAX is using '$BACKEND' instead of 'gpu'."
    echo "Install CUDA-enabled JAX:  pip install --upgrade 'jax[cuda12]'"
    exit 1
fi
echo "==================="

python train.py \
    --phase 2 \
    --conditioning-method film \
    --freeze-trunk false \
    --prompt-style cot_generated
