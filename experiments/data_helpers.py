import logging
from abc import ABC

import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from experiments.data_loaders import DefaultXYDataLoader

logger = logging.getLogger(__name__)


class DataHelper(object):
    """

    Helps to load experimental data as PyTorch data loaders

    """
    train_test_split = 0.8
    train_batch_size = None
    test_batch_size = None
    random_seed = None
    tqdm_cls = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f'Unknown attribute: {k}')

    def get_data_loaders(self):
        raise NotImplementedError()

    @staticmethod
    def get_item_lengths(data_loader: DataLoader, masks_idx):
        """
        Extract the length of data items in data loader (with masks)

        Inspect output with Pandas like this: `pd.Series(lengths).describe()`

        :param data_loader:
        :param masks_idx: Index of mask data in batch
        :return: List of length
        """
        lengths = []

        for batch in data_loader:
            for mask in batch[masks_idx]:
                lengths.append(int(mask.sum()))

        return lengths

    def get_train_test_split(self, df):
        split_at = int(len(df) * self.train_test_split)

        split_df = df.sample(frac=1., random_state=self.random_seed).reset_index(drop=True)

        train_df = split_df[:split_at]
        test_df = split_df[split_at:]

        logger.info(f'Train: {len(train_df)}; Test: {len(test_df)} (ratio: {self.train_test_split})')

        return train_df, test_df

    def get_data_sampler(self, sampler=None, dataset=None, sampler_cls=None):
        """

        Handle different ways to sample data from data loader (Random, sequential, weighted, ..)

        :param sampler:
        :param dataset:
        :param sampler_cls:
        :return:
        """
        if sampler is not None:
            return sampler  # WeightedRandomSampler
        elif sampler_cls is not None:
            return sampler_cls(dataset)  # Sequential or RandomSampler
        else:
            raise ValueError('Either `sampler` or `sampler_cls` must be set!')


class BERTDataHelper(DataHelper):
    """
    For BERT/Transformer specific input (tokenizer, ...)
    """
    doc_a_col = None  # type: str
    doc_b_col = None  # type: str

    tokenizer = None
    bert_model_path = None
    bert_tokenizer_cls = BertTokenizer
    bert_tokenizer_params = {
        'do_lower_case': True,
    }

    negative_sampling_ratio = 1.
    max_seq_length = 512

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = self.bert_tokenizer_cls.from_pretrained(self.bert_model_path, **self.bert_tokenizer_params)

        return self.tokenizer

    def get_joint_token_ids_and_types(self, pairs, token_ids_map):
        """
        Converts document pairs into a joint set of tokens for JointBERT models.

        Token format: [CLS] doc_a [SEP] doc_b [SEP]
        Token type ids: 0 0 0 1 1

        :param pairs: list of tuples with A + B (title or ids depending on keys of token_ids_map)
        :param token_ids_map:
        :return: joint_token_ids (tensor), masks (tensor), token_type_ids (tensor)
        """

        reserved_tokens_count = 3
        max_side_length = int((self.max_seq_length - reserved_tokens_count) / 2)

        joint_ids = []
        token_types = []

        logger.info(f'Joining token pairs with max_side_length={max_side_length}')

        if self.tqdm_cls:
            pairs = self.tqdm_cls(pairs, total=len(pairs), desc='Joining documents')

        for a, b in pairs:
            token_ids_a = token_ids_map[a]
            token_ids_b = token_ids_map[b]

            len_a = len(token_ids_a)
            len_b = len(token_ids_b)

            if len_a > max_side_length and len_b > max_side_length:  # both a too long
                token_ids_a = token_ids_a[:max_side_length]
                token_ids_b = token_ids_b[:max_side_length]
            elif len_a > max_side_length and len_b <= max_side_length:  # a is long, b is short
                token_ids_a = token_ids_a[:max_side_length + max_side_length - len_b]
                token_ids_b = token_ids_b
            elif len_a <= max_side_length and len_b > max_side_length:  # a is short, b is long
                token_ids_a = token_ids_a
                token_ids_b = token_ids_b[:max_side_length + max_side_length - len_a]
            else:
                token_ids_a = token_ids_a
                token_ids_b = token_ids_b

            # joint = [self.get_tokenizer().cls_token_id] + token_ids_a + \
            #         [self.get_tokenizer().sep_token_id] + token_ids_b + [self.get_tokenizer().sep_token_id]
            joint = self.get_tokenizer().build_inputs_with_special_tokens(token_ids_a, token_ids_b)

            joint_ids.append(torch.tensor(joint))

            # [CLS] ids, .. [SEP] ... [SEP]
            # token_types.append(torch.tensor([0] * (2 + len(token_ids_a)) + [1] * (1 + len(token_ids_b))))
            token_types.append(torch.tensor(self.get_tokenizer().create_token_type_ids_from_sequences(token_ids_a, token_ids_b)))

        joint_ids = pad_sequence(joint_ids, batch_first=True, padding_value=self.get_tokenizer().pad_token_id)
        #joint_ids.size()

        masks = torch.tensor([[float(i > 0) for i in ii] for ii in joint_ids])

        token_types = pad_sequence(token_types, batch_first=True, padding_value=1)

        return joint_ids, masks, token_types


    def to_siamese_data_loader(self, df, token_ids_map, batch_size, sampler_cls=None, sampler=None):
        ys = self.get_ys_as_tensor(df)

        doc_ids = df[[self.doc_a_col, self.doc_b_col]].values

        if self.tqdm_cls:
            doc_ids = self.tqdm_cls(doc_ids, total=len(doc_ids), desc='Building tensor data set')

        #self.get_tokenizer()
        token_ids_a = [torch.tensor([self.get_tokenizer().cls_token_id] + token_ids_map[a][:self.max_seq_length - 2] + [
            self.get_tokenizer().sep_token_id]) for a, b in doc_ids]
        token_ids_b = [torch.tensor([self.get_tokenizer().cls_token_id] + token_ids_map[b][:self.max_seq_length - 2] + [
            self.get_tokenizer().sep_token_id]) for a, b in doc_ids]

        # token_ids_a = [torch.tensor([self.get_tokenizer().cls_token_id] + token_ids_map[a][:self.max_seq_length - 2] + [self.get_tokenizer().sep_token_id]) for a, b in doc_ids]
        # token_ids_b = [torch.tensor([self.get_tokenizer().cls_token_id] + token_ids_map[b][:self.max_seq_length - 2] + [self.get_tokenizer().sep_token_id]) for a, b in doc_ids]

        token_ids_a = pad_sequence(token_ids_a, batch_first=True, padding_value=self.get_tokenizer().pad_token_id)
        token_ids_b = pad_sequence(token_ids_b, batch_first=True, padding_value=self.get_tokenizer().pad_token_id)

        masks_a = torch.tensor([[float(i > 0) for i in ii] for ii in token_ids_a])
        masks_b = torch.tensor([[float(i > 0) for i in ii] for ii in token_ids_b])

        # build dataset
        dataset = TensorDataset(
            token_ids_a,
            masks_a,
            token_ids_b,
            masks_b,
            ys)

        return DefaultXYDataLoader(dataset, sampler=self.get_data_sampler(sampler, dataset, sampler_cls), batch_size=batch_size)

    def to_joint_data_loader(self, df, token_ids_map, batch_size, sampler_cls=None, sampler=None):
        ys = self.get_ys_as_tensor(df)

        doc_ids = df[[self.doc_a_col, self.doc_b_col]].values
        joint_ids, masks, token_types = self.get_joint_token_ids_and_types(doc_ids, token_ids_map)

        # build dataset
        dataset = TensorDataset(
            joint_ids,
            masks,
            token_types,
            ys)

        return DefaultXYDataLoader(dataset, sampler=self.get_data_sampler(sampler, dataset, sampler_cls), batch_size=batch_size)


class DocRelDataHelper(object):
    labels = ['employer']  # 'employer' # 'capital' # 'country_of_citizenship' #'educated_at' # 'opposite_of'
    label_col = None
    none_label = 'none'
    label_encoder = None
    labels_integer_encoded = None
    onehot_encoder = None
    labels_onehot_encoded = None

    def get_labels_count(self):
        """
        If "none label" is set, count is increased by one.

        :return:
        """
        if self.none_label:
            return len(self.labels) + 1
        else:
            return len(self.labels)

    def set_label_encoder(self, df):
        self.label_encoder = LabelEncoder()
        # self.labels_integer_encoded = self.label_encoder.fit_transform(list(df[self.label_col].values))
        label_values = list(df[self.label_col].values)

        if self.none_label:
            label_values.append(self.none_label)

        self.labels_integer_encoded = self.label_encoder.fit_transform(label_values)

        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        self.labels_onehot_encoded = self.onehot_encoder.fit_transform(
            self.labels_integer_encoded.reshape(len(self.labels_integer_encoded), 1))

    def is_binary_classification(self):
        return len(self.labels) == 1

    def get_ys_as_tensor(self, df):
        # convert categorical labels into numbers (one hot vectors)
        if self.is_binary_classification():
            return torch.tensor(self.label_encoder.transform(df[self.label_col].values).reshape(len(df), 1)).double()
        else:
            onehot_encoded = self.onehot_encoder.transform(
                self.label_encoder.transform(df[self.label_col].values).reshape(len(df), 1)
            )
            return torch.tensor(onehot_encoded)