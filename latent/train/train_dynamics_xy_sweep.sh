#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

AE_CHECKPOINT="${DYN_AE_CHECKPOINT:-${REPO_ROOT}/latent/train/exp/OGBench/LatentAutoEncoder/sd000_20260623_212844/params_1000000.pkl}"
SAVE_DIR="${DYN_SAVE_DIR:-${REPO_ROOT}/latent/train/exp}"
TRAIN_STEPS="${DYN_TRAIN_STEPS:-1000000}"
WANDB_MODE="${DYN_WANDB_MODE:-online}"
XY_TOLERANCE="${DYN_XY_TOLERANCE:-1.0}"

WEIGHTS=(1 5 10)
SEEDS=(0 1 2)

if [[ ! -f "${AE_CHECKPOINT}" ]]; then
    echo "Autoencoder checkpoint not found: ${AE_CHECKPOINT}" >&2
    exit 1
fi

shopt -s nullglob

for weight in "${WEIGHTS[@]}"; do
    run_group="LatentDynamicsResidualXYw${weight}"
    for seed in "${SEEDS[@]}"; do
        seed_prefix="$(printf 'sd%03d_' "${seed}")"
        completed=(
            "${SAVE_DIR}/OGBench/${run_group}/${seed_prefix}"*/"params_${TRAIN_STEPS}.pkl"
        )
        if (( ${#completed[@]} > 0 )); then
            echo "Skipping weight=${weight}, seed=${seed}; found ${completed[0]}"
            continue
        fi

        echo "Starting weight=${weight}, seed=${seed}, group=${run_group}"
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            python latent/train/train_dynamics.py \
                --ae_checkpoint_path="${AE_CHECKPOINT}" \
                --env_name=antmaze-large-navigate-v0 \
                --save_dir="${SAVE_DIR}" \
                --run_group="${run_group}" \
                --wandb_mode="${WANDB_MODE}" \
                --latent_loss_mode=l2 \
                --prediction_type=residual \
                --xy_weight="${weight}" \
                --xy_tolerance="${XY_TOLERANCE}" \
                --train_steps="${TRAIN_STEPS}" \
                --seed="${seed}"
    done
done

echo "Completed all residual XY-loss sweep runs."
