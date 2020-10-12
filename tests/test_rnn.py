import os
from collections import defaultdict
from unittest import TestCase

import spacy
import torch
from transformers import AutoTokenizer, AutoConfig, RobertaTokenizer, RobertaForSequenceClassification

from acl.__data_prep import load_parscit_file, get_citation_context
from acl.trainer_utils import get_vectors_from_spacy_model
from experiments.environment import get_env
from models.auto_modeling import AutoModelForMultiLabelSequenceClassification
from models.rnn import RNNForMultiLabelSequenceClassification
from trainer_cli import ExperimentArguments


class AutoModelingTest(TestCase):
    env = None

    def setUp(self) -> None:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_WATCH"] = "false"
        self.env = get_env()

        self.cache_dir = '../data/transformers_cache'
        self.sample_text = ' '.join(['Hello world! '] * 10)
        self.num_labels = 5

    def test_rnn_model(self):

        # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path, cache_dir=self.cache_dir)
        # model_config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, cache_dir=self.cache_dir)

        # model = AutoModelForMultiLabelSequenceClassification.from_pretrained(model_name_or_path, config=model_config, cache_dir=self.cache_dir)

        experiment_args = ExperimentArguments('s2_id', 'from_s2_id', 'to_s2_id', 1, 'acl_docrel')
        # label_classes
        spacy_nlp = spacy.load(experiment_args.spacy_model, disable=["tagger", "ner", "textcat"])

        model = RNNForMultiLabelSequenceClassification(
            word_vectors=get_vectors_from_spacy_model(spacy_nlp),
            hidden_size=experiment_args.rnn_hidden_size,
            rnn=experiment_args.rnn_type,
            num_labels=self.num_labels,
            num_layers=experiment_args.rnn_num_layers,
            dropout=experiment_args.rnn_dropout,
        )
        #
        # model.eval()
        #
        # encodings = tokenizer.batch_encode_plus([text], return_tensors='pt')
        #
        # return model(encodings['input_ids']), model, tokenizer
