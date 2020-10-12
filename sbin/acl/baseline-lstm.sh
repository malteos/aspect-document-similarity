#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

. $APP_ROOT/config.sh

export MODEL_NAME=baseline-rnn
export EPOCHS=10
export SPACY_MODEL=~/datasets/spacy/en_glove_6b_300d
export SPACY_MODEL=/Volumes/data/repo/data/spacy/en_glove_6b_300d

for CV_FOLD in 1
do
    python trainer_cli.py --cv_fold $CV_FOLD \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $MODEL_NAME \
        --doc_id_col $DOC_ID_COL \
        --doc_a_col $DOC_A_COL \
        --doc_b_col $DOC_B_COL \
        --nlp_dataset $NLP_DATASET \
        --nlp_cache_dir $NLP_CACHE_DIR \
        --cache_dir $CACHE_DIR \
        --num_train_epochs $EPOCHS \
        --seed $SEED \
        --learning_rate $LR \
        --logging_steps 500 \
        --save_steps 0 \
        --save_total_limit 3 \
        --do_train \
        --save_predictions \
        --spacy_model $SPACY_MODEL \
        --rnn_type lstm \
        --evaluate_during_training \
        --no_cuda
done

export PYTHONUNBUFFERED=""
