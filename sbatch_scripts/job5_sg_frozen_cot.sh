#!/bin/bash
#SBATCH --job-name=sg_frozen_cot
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/job5_sg_frozen_cot_%j.out
#SBATCH --error=slurm_logs/job5_sg_frozen_cot_%j.err
#SBATCH --time=12:00:00

# ── Job 5: spectral_gating, frozen trunk, cot_generated ──
# All 6 PDEs. First run generates TinyLlama CoT to cache/cot_generated/; later runs reuse it.

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
    --conditioning-method spectral_gating \
    --freeze-trunk true \
    --prompt-style cot_generated
