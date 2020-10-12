import importlib.util
import os
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional, Any

import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.metrics import classification_report
from transformers import DataCollator, PreTrainedTokenizer
from transformers import EvalPrediction

from experiments.utils import flatten


def get_label_classes_from_nlp_dataset(cls_path: str, attr_name='LABEL_CLASSES') -> List[str]:
    if not cls_path.endswith('.py'):
        raise ValueError('data path must point to .py-file')

    if not cls_path.startswith('./'):
        raise ValueError('Must be relative path')

    # Make absolute path from app root
    cls_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), cls_path[2:])

    # Get file name, remove .py
    cls_name = cls_path[:-3].split('/')[-1]

    spec = importlib.util.spec_from_file_location(cls_name, cls_path)
    dataset_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(dataset_module)

    if hasattr(dataset_module, attr_name):
        return getattr(dataset_module, attr_name)
    else:
        raise ValueError(f'dataset module does not have attribute: {attr_name}')


def get_vectors_from_spacy_model(spacy_nlp):
    unk_token_vector = np.zeros((1, spacy_nlp.vocab.vectors.shape[1]))
    sep_token_vector = np.ones((1, spacy_nlp.vocab.vectors.shape[1]))
    return np.concatenate((spacy_nlp.vocab.vectors.data,
                                              unk_token_vector,
                                              sep_token_vector), axis=0)


class DocRelTrainerHelper(object):
    def __init__(self,
                 id2doc: Dict[str, Dict],
                 label_classes: List[str],
                 doc_a_col: str,
                 doc_b_col: str,
                 label_col: str,
                 text_from_doc_func: Callable,
                 max_length=512,
                 spacy_nlp: Optional[Any] = None,
                 transformers_tokenizer: Optional[PreTrainedTokenizer] = None,
                 classification_threshold: float = 0.):
        self.id2doc = id2doc
        self.transformers_tokenizer = transformers_tokenizer
        self.spacy_nlp = spacy_nlp
        self.label_classes = label_classes
        self.doc_a_col = doc_a_col
        self.doc_b_col = doc_b_col
        self.label_col = label_col
        self.max_length = max_length
        self.classification_threshold = classification_threshold
        self.text_from_doc_func = text_from_doc_func

        if self.transformers_tokenizer and (self.transformers_tokenizer.max_len is None or self.transformers_tokenizer.max_len < 1):
            raise ValueError('Tokenizer max_length is not set!')

        if self.spacy_nlp:
            # Extend vocabulary with UNK + SEP token
            self.spacy_unk_token_id = len(self.spacy_nlp.vocab.vectors) + 0
            self.spacy_sep_token_id = len(self.spacy_nlp.vocab.vectors) + 1
        else:
            self.spacy_unk_token_id = self.spacy_sep_token_id = None

    def convert_to_features(self, batch):
        if self.transformers_tokenizer:
            return self.convert_to_features_transformers(batch)
        elif self.spacy_nlp:
            return self.convert_to_features_spacy(batch)
        else:
            raise ValueError('Neither Transformers tokenizer nor Spacy is set!')

    def convert_to_features_spacy(self, batch):
        snlp = self.spacy_nlp
        label_encodings = []
        input_ids = []
        attention_masks = []

        for from_id, to_id, label in zip(batch[self.doc_a_col], batch[self.doc_b_col], batch[self.label_col]):
            if from_id not in self.id2doc:
                raise ValueError(f'Document not found. from_id={from_id}; label={label}')
            elif to_id not in self.id2doc:
                raise ValueError(f'Document not found. to_id={to_id}; label={label}')

            from_doc = self.id2doc[from_id]
            from_tokens = snlp(self.text_from_doc_func(from_doc))[:np.floor(self.max_length / 2)]
            from_token_ids = [snlp.vocab.vectors.key2row[t.norm] if t.has_vector and t.norm in snlp.vocab.vectors.key2row else self.spacy_unk_token_id for t in from_tokens]

            to_doc = self.id2doc[to_id]
            to_tokens = snlp(self.text_from_doc_func(to_doc))[:np.floor(self.max_length / 2)]
            to_token_ids = [snlp.vocab.vectors.key2row[t.norm] if t.has_vector and t.norm in snlp.vocab.vectors.key2row else self.spacy_unk_token_id for t in
                              to_tokens]

            # Join with SEP token
            token_ids = from_token_ids + [self.spacy_sep_token_id] + to_token_ids
            token_ids = token_ids[:self.max_length]

            attention_mask = np.zeros(self.max_length)
            attention_mask[list(range(len(token_ids)))] = 1.

            # Zero-padding
            if len(token_ids) < self.max_length:
                token_ids += [0] * (self.max_length - len(token_ids))

            one_hot_encoded_label = np.zeros(len(self.label_classes))
            one_hot_encoded_label[[self.label_classes.index(l) for l in label]] = 1.

            # To list
            attention_masks.append(attention_mask.tolist())
            input_ids.append(token_ids)
            label_encodings.append(one_hot_encoded_label)

        encodings = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'token_type_ids': [[0] * self.max_length] * len(input_ids),
            'labels': label_encodings,
        }

        return encodings

    def convert_to_features_transformers(self, batch):
        text_pairs = []
        label_encodings = []

        for from_id, to_id, label in zip(batch[self.doc_a_col], batch[self.doc_b_col], batch[self.label_col]):
            if from_id not in self.id2doc:
                raise ValueError(f'Document not found. from_id={from_id}; label={label}')
            elif to_id not in self.id2doc:
                raise ValueError(f'Document not found. to_id={to_id}; label={label}')
            else:
                from_doc = self.id2doc[from_id]
                to_doc = self.id2doc[to_id]

                text_pairs.append((
                    self.text_from_doc_func(from_doc), self.text_from_doc_func(to_doc)
                ))

            one_hot_encoded_label = np.zeros(len(self.label_classes))
            one_hot_encoded_label[[self.label_classes.index(l) for l in label]] = 1.

            label_encodings.append(one_hot_encoded_label)

        input_encodings = self.transformers_tokenizer.batch_encode_plus(
            text_pairs,
            pad_to_max_length=True,
            truncation_strategy='longest_first',
            return_token_type_ids=True,
            return_attention_masks=True,
            max_length=self.max_length
        )

        # RoBERTa does not make use of token type ids, therefore a list of zeros is returned.
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'token_type_ids': input_encodings['token_type_ids'],
            'labels': label_encodings,
        }

        # if 'token_type_ids' in input_encodings:
        #     input_encodings['token_type_ids'] = input_encodings['token_type_ids']

        return encodings

    def compute_metrics(self, p: EvalPrediction) -> Dict:
        predicted_labels = np.where(p.predictions > self.classification_threshold, 1., 0.)

        return flatten(classification_report(
                y_true=p.label_ids,
                y_pred=predicted_labels,
                target_names=self.label_classes,
                zero_division=0,
                output_dict=True))

    def get_df_from_predictions(self, relations_dataset, docs_dataset, predictions, exclude_columns: List=None):
        if exclude_columns is None:
            exclude_columns = []

        # To dataframe with IDs ...
        true_dict = {'true_' + label: predictions.label_ids[:, idx] for idx, label in enumerate(self.label_classes)}
        predictions_dict = {'predicted_' + label: predictions.predictions[:, idx] for idx, label in
                            enumerate(self.label_classes)}
        predictions_label_lists = [
            [label for idx, label in enumerate(self.label_classes) if item[idx] > self.classification_threshold] for item in
            predictions.predictions]

        # Document meta data
        from_dict = {
            'from_' + col: [self.id2doc[s2_id][col] if s2_id in self.id2doc else None for s2_id in relations_dataset[self.doc_a_col]]
            for col in docs_dataset.column_names if col not in exclude_columns}
        to_dict = {'to_' + col: [self.id2doc[s2_id][col] if s2_id in self.id2doc else None for s2_id in relations_dataset[self.doc_b_col]]
                   for col in docs_dataset.column_names if col not in exclude_columns}

        df_dict = {}
        df_dict.update(from_dict)
        df_dict.update(to_dict)

        df_dict.update({
            # Labels
            'true': [','.join(label_list) for label_list in relations_dataset[self.label_col]],
            'predicted': [','.join(label_list) for label_list in predictions_label_lists],
        })
        df_dict.update(true_dict)
        df_dict.update(predictions_dict)

        return pd.DataFrame.from_dict(df_dict)


@dataclass
class DocRelDataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """

        input_ids = torch.stack([example['input_ids'] for example in batch])
        token_type_ids = torch.stack([example['token_type_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        labels = torch.stack([example['labels'].squeeze() for example in batch])

        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
        }

        return model_kwargs


def get_non_empty_text_from_doc(doc) -> str:
    """
    Build document text from title + abstract

    :param doc: S2 paper
    :return: Document text
    """

    text = ''

    if 'title' in doc:
        text += doc['title']

        if doc['abstract']:
            text += '\n' + doc['abstract']

    if len(text) == 0:
        # Ensure text is at least one char to make tokenizers work.
        text = ' '

    return text
