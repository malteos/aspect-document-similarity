#!/usr/bin/env bash

export LR=2e-5  # 5e-05 # LEARNING_RATE = 2e-5  # 2e-6 does not work (?)
export EPOCHS=4  # or 4?
export SEED=0
export NLP_CACHE_DIR=./data/nlp_cache
export CACHE_DIR=./data/trainer_cache

export OUTPUT_DIR=./output/cord19_docrel/folds
export DOC_ID_COL=doi
export DOC_A_COL=from_doi
export DOC_B_COL=to_doi
export NLP_DATASET=./datasets/cord19_docrel/cord19_docrel.py

# wandb
export WANDB_API_KEY=
export WANDB_PROJECT=
