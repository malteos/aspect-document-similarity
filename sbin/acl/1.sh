#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

export APP_ROOT=$(dirname "$0")

. $APP_ROOT/config.sh

# models: albert-base-v1   bert-base-german-cased        biobert-v1-1             longformer-base-4096.tar.gz   pytorch
# scibert-scivocab-uncased
#albert-base-v2   bert-base-multilingual-cased  distilbert-base-uncased
# longformer-large-4096         roberta-base   xlnet-base-cased
#bert-base-cased  bert-large-cased
# longformer-base-4096     longformer-large-4096.tar.gz
# roberta-large

export MODEL_NAME=bert-base-cased
export MODEL_NAME=bert-large-cased

export MODEL_NAME=roberta-base
export MODEL_NAME=longformer-base-4096
export MODEL_NAME=xlnet-base

export CV_FOLD=1

# longformer
export EVAL_BATCH_SIZE=4
export TRAIN_BATCH_SIZE=4

# large
export EVAL_BATCH_SIZE=4
export TRAIN_BATCH_SIZE=2

# bert-base
export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=8

# xlnet-base
export EVAL_BATCH_SIZE=12
export TRAIN_BATCH_SIZE=6



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
        --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
        --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
        --learning_rate $LR \
        --logging_steps 100 \
        --save_steps 0 \
        --save_total_limit 3 \
        --do_train \
        --save_predictions
done

######

export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=8

for MODEL_NAME in "bert-base-cased" "scibert-scivocab-uncased" "roberta-base" "xlnet-base-cased" "google/electra-base-discriminator" "deepset/covid_bert_base"
do
    echo $MODEL_NAME
    export CV_FOLD=1
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
        --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
        --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
        --learning_rate $LR \
        --logging_steps 100 \
        --save_steps 0 \
        --save_total_limit 3 \
        --do_train \
        --save_predictions
done


export PYTHONUNBUFFERED=""

