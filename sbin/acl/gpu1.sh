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

export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=8

export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=8

# serv 9212; gpu 0
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_NAME="bert-base-cased"

echo $MODEL_NAME
for CV_FOLD in 1 2 3 4
do
    echo $CV_FOLD
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

# serv 9212; gpu 1
export CUDA_VISIBLE_DEVICES=1
export MODEL_NAME="scibert-scivocab-uncased"
echo $MODEL_NAME
for CV_FOLD in 1 2 3 4
do
    echo $CV_FOLD
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

# serv 9212; gpu 1
export CUDA_VISIBLE_DEVICES=1

export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=12
export EPOCHS=8
export CV_FOLD=1
export LR=1e-5
export RNN_NUM_LAYERS=2
export RNN_HIDDEN_SIZE=100
export RNN_DROPOUT=0.1
export SPACY_MODEL=./output/acl_docrel/spacy/en_acl_fasttext_300d
export MODEL_NAME=baseline-rnn__fasttext__custom

for CV_FOLD in 1 2 3 4
do
    echo $CV_FOLD
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
        --spacy_model $SPACY_MODEL \
        --rnn_type lstm \
        --rnn_num_layers $RNN_NUM_LAYERS \
        --rnn_hidden_size $RNN_HIDDEN_SIZE \
        --rnn_dropout $RNN_DROPOUT \
        --do_train \
        --save_predictions
done



######
######
######
######


# serv 9200; gpu 2
export CUDA_VISIBLE_DEVICES=2
export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=8
export MODEL_NAME="roberta-base"
for CV_FOLD in 1 2 3 4
do
    echo $CV_FOLD
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

# serv 9200; gpu 3
export CUDA_VISIBLE_DEVICES=3
export EVAL_BATCH_SIZE=12
export TRAIN_BATCH_SIZE=6
export MODEL_NAME="xlnet-base-cased"
for CV_FOLD in 1 2 3 4
do
    echo $CV_FOLD
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


# serv 9200; gpu 4
export CUDA_VISIBLE_DEVICES=4
export EVAL_BATCH_SIZE=12
export TRAIN_BATCH_SIZE=8
export MODEL_NAME="google/electra-base-discriminator"
for CV_FOLD in 1 2 3 4
do
    echo $MODEL_NAME
    echo $CV_FOLD
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


# serv 9200; gpu 5
export CUDA_VISIBLE_DEVICES=5
export EVAL_BATCH_SIZE=16
export TRAIN_BATCH_SIZE=8
export MODEL_NAME="deepset/covid_bert_base"
for CV_FOLD in 1 2 3 4
do
    echo $MODEL_NAME
    echo $CV_FOLD
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


#####

# serv 9200; gpu 3
export CUDA_VISIBLE_DEVICES=2,4,5
export EVAL_BATCH_SIZE=12
export TRAIN_BATCH_SIZE=6
export MODEL_NAME="xlnet-base-cased"
for CV_FOLD in 4
do
    echo $CV_FOLD
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

