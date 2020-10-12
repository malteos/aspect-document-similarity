#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

. $APP_ROOT/config.sh

export MODEL_NAME=baseline-rnn
export EPOCHS=10
export SPACY_MODEL=~/datasets/spacy/en_glove_6b_300d

export MODEL_NAME=baseline-rnn__fasttext
export SPACY_MODEL=~/datasets/spacy/en_fasttext_wiki-news-300d-1m

export MODEL_NAME=baseline-rnn__fasttext__custom
export SPACY_MODEL=./output/acl_docrel/spacy/en_acl_fasttext_300d

export EVAL_BATCH_SIZE=12
export TRAIN_BATCH_SIZE=8

export EPOCHS=10
export CV_FOLD=1
export LR=1e-5
export RNN_NUM_LAYERS=2
export RNN_HIDDEN_SIZE=100
export RNN_DROPOUT=0.1

# [1] Reimers, N. and Gurevych, I. 2016. Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks. (2016).
# -
# A value of about 100 for each LSTM-network appears to be a good rule of thumb for the tested tasks
# -
# For tasks with small training sets appears a mini-batch size of 8 a robust selection.
# For tasks with larger training sets appears a mini-batch size of 32 a robust selection.
# -
# Except for the reduced POS tagging task, two BiLSTM-layers produced the best re- sults.
# -
# Variational dropout was on all tasks superior to no-dropout or naive dropout.
# Applying dropout along the vertical as well as the recurrent dimension achieved on all benchmark tasks the best result.
# 0.1 => same as in transformers

for CV_FOLD in 1 2 3 4
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
        --rnn_num_layers $RNN_NUM_LAYERS \
        --rnn_hidden_size $RNN_HIDDEN_SIZE \
        --rnn_dropout $RNN_DROPOUT \
        --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
        --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
        --evaluate_during_training
done

export PYTHONUNBUFFERED=""
