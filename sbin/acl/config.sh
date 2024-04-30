#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=1

export LR=2e-5  # 5e-05 # LEARNING_RATE = 2e-5  # 2e-6 does not work (?)
export EPOCHS=4  # or 4?
export SEED=0
export NLP_CACHE_DIR=./data/nlp_cache
export CACHE_DIR=./data/trainer_cache

export OUTPUT_DIR=./output/acl_docrel/folds
export DOC_ID_COL=s2_id
export DOC_A_COL=from_s2_id
export DOC_B_COL=to_s2_id
export NLP_DATASET=./datasets/acl_docrel/acl_docrel.py

# wandb
export WANDB_API_KEY=
export WANDB_PROJECT=
