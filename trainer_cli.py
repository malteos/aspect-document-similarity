import dataclasses
import hashlib
import json
import os
import logging

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nlp
import spacy
import torch
import wandb
from nlp import load_dataset
from torch import nn
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, PreTrainedModel)
from transformers.trainer import is_wandb_available

from acl.trainer_utils import DocRelTrainerHelper, DocRelDataCollator, get_label_classes_from_nlp_dataset, \
    get_vectors_from_spacy_model, get_non_empty_text_from_doc
from acl.utils import get_text_from_doc
from datasets.acl_docrel.acl_docrel import get_train_split, get_test_split, LABEL_CLASSES
from experiments.environment import get_env
from models.auto_modeling import AutoModelForMultiLabelSequenceClassification
from models.bert import BertForMultiLabelSequenceClassification
from models.rnn import RNNForMultiLabelSequenceClassification

logger = logging.getLogger(__name__)


@dataclass
class ExperimentArguments:
    """
    Arguments for our experimental setup.
    """
    doc_id_col: str = field(
        metadata={"help": "Column in which document ID is stored"}
    )
    doc_a_col: str = field(
        metadata={"help": "Column name for document A"}
    )
    doc_b_col: str = field(
        metadata={"help": "Column name for document B"}
    )
    cv_fold: int = field(
        metadata={"help": "Cross validation fold"}
    )
    nlp_dataset: str = field(
        metadata={"help": "Name or path for dataset downloaded with huggingface's nlp"}
    )
    nlp_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Datasets downloaded with huggingface's nlp are cached in this directory"}
    )
    label_col: Optional[str] = field(
        default="label",
        metadata={"help": "Column name for label"}
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of input sequence"}
    )
    classification_threshold: Optional[float] = field(
        default=0.,
        metadata={"help": "Predicted probability must be >= than this threshold for classification"}
    )
    save_predictions: bool = field(
        default=False,
        metadata={"help": "Generate predictions after training and save them to disk"}
    )
    spacy_model: Optional[str] = field(
        default=None,
        metadata={"help": "Name or path to Spacy model (only used for RNN baseline)"}
    )
    rnn_type: Optional[str] = field(
        default='lstm',
        metadata={"help": "RNN type (lstm or gru)"}
    )
    rnn_hidden_size: Optional[int] = field(
        default=100,
        metadata={"help": "RNN size of hidden layer"}
    )
    rnn_num_layers: Optional[int] = field(
        default=1,
        metadata={"help": "RNN Number of hidden layers"}
    )
    rnn_dropout: Optional[float] = field(
        default=0.,
        metadata={"help": "RNN drop out probability"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    def get_model_name(self):
        return self.model_name_or_path.split('/')[-1]


def main():
    # Auto-environment
    env = get_env()

    parser = HfArgumentParser((ModelArguments, TrainingArguments, ExperimentArguments))
    model_args, training_args, experiment_args = parser.parse_args_into_dataclasses()

    # Adjust output with folds and model name
    training_args.output_dir = os.path.join(training_args.output_dir, str(experiment_args.cv_fold), model_args.get_model_name())

    # Model path from env
    if not os.path.exists(model_args.model_name_or_path) and os.path.exists(os.path.join(env['bert_dir'], model_args.model_name_or_path)):
        model_args.model_name_or_path = os.path.join(env['bert_dir'], model_args.model_name_or_path)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Dataset args
    label_classes = get_label_classes_from_nlp_dataset(experiment_args.nlp_dataset)
    columns = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']

    # Build dataset for splits
    train_ds = load_dataset(experiment_args.nlp_dataset,
                            name='relations',
                            cache_dir=experiment_args.nlp_cache_dir,
                            split=get_train_split(experiment_args.cv_fold))
    test_ds = load_dataset(experiment_args.nlp_dataset,
                           name='relations',
                           cache_dir=experiment_args.nlp_cache_dir,
                           split=get_test_split(experiment_args.cv_fold))
    docs_ds = load_dataset(experiment_args.nlp_dataset,
                           name='docs',
                           cache_dir=experiment_args.nlp_cache_dir,
                           split=nlp.Split('docs'))

    # Build ID => Doc mapping
    doc_id2doc = {doc[experiment_args.doc_id_col]: doc for doc in docs_ds}

    if model_args.model_name_or_path.startswith('baseline-rnn'):
        # Load Spacy as tokenizer
        spacy_nlp = spacy.load(experiment_args.spacy_model, disable=["tagger", "ner", "textcat"])

        # Baseline models
        model = RNNForMultiLabelSequenceClassification(
            word_vectors=get_vectors_from_spacy_model(spacy_nlp),
            hidden_size=experiment_args.rnn_hidden_size,
            rnn=experiment_args.rnn_type,
            num_labels=len(label_classes),
            num_layers=experiment_args.rnn_num_layers,
            dropout=experiment_args.rnn_dropout,
        )
        tokenizer = None

    else:
        # Load pretrained Transformers models and tokenizers
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(label_classes),
            cache_dir=model_args.cache_dir
        )

        # No need for spacy
        spacy_nlp = None

        if 'longformer' in model_args.model_name_or_path:
            # TVM: a custom CUDA kernel implementation of our sliding window attention (works only on GPU)
            model_config.attention_mode = 'tvm'

            # override tokenizer name if not set
            if model_args.tokenizer_name is None:
                roberta_path = os.path.join(env['bert_dir'], 'roberta-base')
                model_args.tokenizer_name = roberta_path if os.path.exists(roberta_path) else 'roberta-base'

                logger.info(f'Overriding tokenizer: {model_args.tokenizer_name}')

            # override max length
            experiment_args.max_length = 4096

        model = AutoModelForMultiLabelSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            cache_dir=model_args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        # Set token limit if defined by model (for Longformer)
        if model.config.max_position_embeddings > 0:
            tokenizer.model_max_length = model.config.max_position_embeddings

    # Init helper
    dpt = DocRelTrainerHelper(
        id2doc=doc_id2doc,
        transformers_tokenizer=tokenizer,
        spacy_nlp=spacy_nlp,
        label_classes=label_classes,
        doc_a_col=experiment_args.doc_a_col,
        doc_b_col=experiment_args.doc_b_col,
        label_col=experiment_args.label_col,
        text_from_doc_func=get_non_empty_text_from_doc,
        classification_threshold=experiment_args.classification_threshold,
        max_length=experiment_args.max_length,
    )

    logger.info('Converting to features (doc mapping, tokenize, ...)')

    # Build hash from settings for caching
    data_settings_hash = hashlib.md5(
        dataclasses.asdict(experiment_args).__str__().encode("utf-8") +
        dataclasses.asdict(model_args).__str__().encode("utf-8")).hexdigest()

    train_ds = train_ds.map(
        dpt.convert_to_features,
        batched=True,
        load_from_cache_file=True,
        cache_file_name=os.path.join(experiment_args.nlp_cache_dir, "cache-train-" + data_settings_hash + ".arrow")
    )
    train_ds.set_format(type='torch', columns=columns)

    test_ds = test_ds.map(
        dpt.convert_to_features,
        batched=True,
        load_from_cache_file=True,
        cache_file_name=os.path.join(experiment_args.nlp_cache_dir, "cache-test-" + data_settings_hash + ".arrow")
    )
    test_ds.set_format(type='torch', columns=columns)

    # Load models weights (when no training but predictions)
    model_weights_path = os.path.join(training_args.output_dir, 'pytorch_model.bin')

    if not training_args.do_train and experiment_args.save_predictions:
        logger.info(f'Loading existing model weights from disk: {model_weights_path}')
        if os.path.exists(model_weights_path):
            model.load_state_dict(torch.load(model_weights_path))
        else:
            logger.error('Weights files does not exist!')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DocRelDataCollator(),
        prediction_loss_only=False,
        compute_metrics=dpt.compute_metrics,
    )

    # Log additional (to Weights & Baises)
    if is_wandb_available():
        wandb.config.update(dataclasses.asdict(experiment_args))
        wandb.config.update(dataclasses.asdict(model_args))

    if training_args.do_train:
        logger.info('Training started...')

        trainer.train()

        if isinstance(model, PreTrainedModel):
            trainer.save_model()
        elif isinstance(model, nn.Module):  # RNN model
            torch.save(model.state_dict(), model_weights_path)

    if experiment_args.save_predictions:
        logger.info('Predicting...')

        predictions = trainer.predict(test_ds)

        df = dpt.get_df_from_predictions(test_ds, docs_ds, predictions, exclude_columns=['abstract'])

        # Save results to disk
        df.to_csv(os.path.join(training_args.output_dir, 'results.csv'), index=False)
        json.dump(predictions.metrics, open(os.path.join(training_args.output_dir, 'metrics.json'), 'w'))

    logger.info('Done')


if __name__ == '__main__':
    main()
