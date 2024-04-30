import os
from collections import defaultdict
from unittest import TestCase

import torch
from transformers import AutoTokenizer, AutoConfig, RobertaTokenizer, RobertaForSequenceClassification

from acl.__data_prep import load_parscit_file, get_citation_context
from experiments.environment import get_env
from models.auto_modeling import AutoModelForMultiLabelSequenceClassification


class AutoModelingTest(TestCase):
    env = None

    def setUp(self) -> None:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_WATCH"] = "false"
        self.env = get_env()

        self.cache_dir = '../data/transformers_cache'
        self.sample_text = ' '.join(['Hello world! '] * 10)
        self.num_labels = 5

    def forward_model(self, model_name_or_path, text, tokenizer_name=None):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path, cache_dir=self.cache_dir)
        model_config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, cache_dir=self.cache_dir)

        model = AutoModelForMultiLabelSequenceClassification.from_pretrained(model_name_or_path, config=model_config, cache_dir=self.cache_dir)

        model.eval()

        encodings = tokenizer.batch_encode_plus([text], return_tensors='pt')

        return model(encodings['input_ids']), model, tokenizer

    def test_bert_auto(self):
        model_name_or_path = self.env['bert_dir'] + '/bert-base-cased'
        out, model, tokenizer = self.forward_model(model_name_or_path, self.sample_text)

        print(out)
        print(type(model))

        print(model.config.max_position_embeddings)

    def test_distilbert_auto(self):
        model_name_or_path = self.env['bert_dir'] + '/distilbert-base-uncased'
        out, model, tokenizer = self.forward_model(model_name_or_path, self.sample_text)

        print(out)
        print(type(model))


        print(model.config.max_position_embeddings)


    def test_xlnet_auto(self):
        model_name_or_path = 'xlnet-base-cased'
        out, model, tokenizer = self.forward_model(model_name_or_path, self.sample_text)

        print(out)
        print(type(model))

        self.assertEqual(self.num_labels, out[0].shape[1])

        print(model.config.max_position_embeddings)
        print(tokenizer.model_max_length)

    def test_roberta_auto(self):
        model_name_or_path = 'roberta-base'
        out, model, tokenizer = self.forward_model(model_name_or_path, self.sample_text)

        print(out)
        print(type(model))

        self.assertEqual(self.num_labels, out[0].shape[1])

        print(model.roberta)
        print(model.config.max_position_embeddings)
        # model.save_pretrained(self.cache_dir)

    def test_roberta_manual(self):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=self.cache_dir)
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', cache_dir=self.cache_dir)

        encodings = tokenizer.batch_encode_plus(['foo bar'], return_tensors='pt')

        print(model(encodings['input_ids']))

    def test_longformer_auto(self):
        model_name_or_path = 'longformer-base-4096'
        out, model, tokenizer = self.forward_model(model_name_or_path, self.sample_text, 'roberta-base')

        print(out)
        print(type(model))

        self.assertEqual(self.num_labels, out[0].shape[1])

        print(model.roberta)
        print(model.config.max_position_embeddings)
        # model.save_pretrained(self.cache_dir)