#!/bin/bash          

ENV_NAME="FetchPickAndPlace-v1"
LOG_DIR="logs"
MODEL_DIR="models"
PLOT_DIR="plots"

TRAIN_TIMESTEPS="1000000"
EVAL_TIMESTEPS="100000"

RESULTS_DIR="FetchPickAndPlace-results"

python train_dqn.py "${ENV_NAME}" "${TRAIN_TIMESTEPS}" "${LOG_DIR}" "${MODEL_DIR}" "${PLOT_DIR}"

python evaluate_dqn.py "${ENV_NAME}" "${TRAIN_TIMESTEPS}" "${EVAL_TIMESTEPS}" "${MODEL_DIR}" "${RESULTS_DIR}"