import logging

import gensim
import torch
from torch import nn as nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

from models import ExperimentModel, get_mlp, get_concat


logger = logging.getLogger(__name__)


"""

BErtModel


        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.



"""

class JointBERT(ExperimentModel):
    """

    Text pair is joint with a [SEP] token.

    This is equal to standard text classification.

    TODO set token_type_ids. See https://github.com/huggingface/transformers/blob/fbaf05bd92249b6dd961f5f8d60eb0892c541ac8/transformers/modeling_bert.py#L524

    """

    def __init__(self, bert_model_path, hidden_dim=768, dropout=0.1, mlp_dim=512, mlp_layers_count=1, labels_count=1, bert_cls=None, prob='auto'):
        super().__init__()

        self.labels_count = labels_count
        self.embedding_dim = hidden_dim  # size of bert representations
        self.bert_cls = bert_cls if bert_cls else BertModel
        self.bert = self.bert_cls.from_pretrained(bert_model_path, output_hidden_states=False, output_attentions=False)

        # distilbert
        self.distil_pooler = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.mlp = get_mlp(input_dim=self.embedding_dim,
                           output_dim=self.labels_count,
                           hidden_dim=mlp_dim,
                           hidden_layers_count=mlp_layers_count,
                           activation_cls=nn.ReLU)

        self.prob = self.get_classification_probability_layer(prob)

    def forward(self, token_ids, masks, token_type_ids):
        # TODO other pooling strategies
        bert_out = self.bert(token_ids, attention_mask=masks, token_type_ids=token_type_ids)

        if len(bert_out) == 1:
            hidden_state = bert_out[0]  # DistilBert or XLNet
            pooler_output = hidden_state[:, 0]                    # (bs, dim)
            pooler_output = self.distil_pooler(pooler_output)   # (bs, dim)
            pooler_output = nn.ReLU()(pooler_output)             # (bs, dim)
        elif len(bert_out) == 2:
            last_hidden_state, pooler_output = bert_out
        else:
            raise ValueError('Transformer output has invalid size')


        dropout_output = self.dropout(pooler_output)

        output = self.mlp(dropout_output)

        if self.prob:
            return self.prob(output)

        return output


class SiameseBERT(ExperimentModel):
    """

    Siamese BERT
    - two identical BERT sub-networks
    - various concat modes
    - MLP on top

    """
    def __init__(self, bert_model_path, hidden_dim=768, dropout=0.1, mlp_dim=512, mlp_layers_count=1, labels_count=1, bert_cls=None, concat='simple', prob='auto'):
        super().__init__()
        self.labels_count = labels_count

        self.embedding_dim = hidden_dim  # size of bert representations
        self.bert_cls = bert_cls if bert_cls else BertModel

        self.bert = self.bert_cls.from_pretrained(bert_model_path, output_hidden_states=False, output_attentions=False)
        self.dropout = nn.Dropout(dropout)

        # distilbert
        self.distil_pooler = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Concat mode
        self.concat = concat
        self.concat_func, self.concat_dim = get_concat(concat, self.embedding_dim)

        self.mlp = get_mlp(input_dim=self.concat_dim,
                   output_dim=self.labels_count,
                   hidden_dim=mlp_dim,
                   hidden_layers_count=mlp_layers_count,
                   activation_cls=nn.ReLU)

        # TODO fill linear layers
        # nn.init.xavier_normal_(self.classifier.weight)
        # Fills the input Tensor with values according to the method described in “Understanding the difficulty of training deep feedforward neural networks” - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
        # kaiming_normal_
        # Fills the input Tensor with values according to the method described in “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015), using a normal distribution.

        self.prob = self.get_classification_probability_layer(prob)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward_document(self, input_ids, masks):
        """
        Sub-network part

        :param input_ids:
        :param masks:
        :return:
        """

        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        bert_out = self.bert(input_ids, attention_mask=masks)

        if len(bert_out) == 1:
            hidden_state = bert_out[0]  # DistilBert
            pooler_output = hidden_state[:, 0]                    # (bs, dim)
            pooler_output = self.distil_pooler(pooler_output)   # (bs, dim)
            pooler_output = nn.ReLU()(pooler_output)             # (bs, dim)
        else:
            last_hidden_state, pooler_output = bert_out

        # outputs = self.bert(tokens, attention_mask=masks)  # sequence_output, pooled_output, (hidden_states), (attentions)
        #print(outputs)

        dropout_output = self.dropout(pooler_output)

        # last_hidden_states = [0][:, -1]
        # dropout_output = self.dropout(last_hidden_states)

        return dropout_output

    def forward(self, tokens_a, masks_a, tokens_b, masks_b):
        a = self.forward_document(tokens_a, masks_a)
        b = self.forward_document(tokens_b, masks_b)

        concat_output = self.concat_func(a, b)

        output = self.mlp(concat_output)

        if self.prob:
            return self.prob(output)

        return output


class SiameseLongBERT(ExperimentModel):
    """

    inspired by https://github.com/AndriyMulyar/bert_document_classification

    """
    def __init__(self, bert_model_path, hidden_dim=768, dropout=0.1, mlp_dim=512, mlp_layers_count=1, labels_count=1, bert_cls=None, concat='simple', prob='auto'):
        super().__init__()
        self.labels_count = labels_count

        self.embedding_dim = hidden_dim  # size of bert representations
        self.bert_cls = bert_cls if bert_cls else BertModel

        self.bert = self.bert_cls.from_pretrained(bert_model_path, output_hidden_states=False)
        self.dropout = nn.Dropout(dropout)

        self.lstm = LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Concat mode
        self.concat = concat
        self.concat_func, self.concat_dim = get_concat(concat, self.embedding_dim)

        self.mlp = get_mlp(input_dim=self.concat_dim,
                   output_dim=self.labels_count,
                   hidden_dim=mlp_dim,
                   hidden_layers_count=mlp_layers_count,
                   activation_cls=nn.ReLU)

        self.prob = self.get_classification_probability_layer(prob)

    def forward_document(self, tokens, masks, chunks):
        """


        :param tokens: size (batch_size, chunks=20, max_seq_length=512)
        :param masks: size (batch_size, chunks=20, max_seq_length=512)
        :param chunks: length of document, i.e. number of chunks a document has
        :return:
        """

        batch_size = tokens.size()[0]
        chunk_size = tokens.size()[1]

        bert_out = torch.zeros((batch_size, chunk_size, self.embedding_dim,), device=self.get_single_device())  # batch_size x chunks x bert_size

        # BERT over all chunks
        with torch.no_grad():
            for chunk_idx in range(chunk_size):
                bert_chunk_out = self.bert(tokens[:, chunk_idx, :], attention_mask=masks[:, chunk_idx, :])[0][:, -1]  # input
                bert_chunk_out = self.dropout(bert_chunk_out)

                bert_out[:, chunk_idx] = bert_chunk_out

        # pack
        packed_bert = pack_padded_sequence(bert_out, lengths=chunks, batch_first=True, enforce_sorted=False)

        packed_lstm_out, (h, c) = self.lstm(packed_bert)

        # unpack sequences
        lstm_out, lstm_out_lengths = pad_packed_sequence(packed_lstm_out, batch_first=True)

        last_lstm_out = lstm_out[:, -1]  # TODO or use last hidden state?

        return last_lstm_out

    def forward(self, tokens_a, masks_a, chunks_a, tokens_b, masks_b, chunks_b):
        a = self.forward_document(tokens_a, masks_a, chunks_a)
        b = self.forward_document(tokens_b, masks_b, chunks_b)

        concat_output = self.concat_func(a, b)

        output = self.mlp(concat_output)

        if self.prob:
            return self.prob(output)

        return output


class SiameseEmbedding(ExperimentModel):
    def __init__(self, embeddings=None, embeddings_path=None, labels_count=1, mlp_dim=512, mlp_layers_count=1, concat='simple', prob='auto'):
        super().__init__()

        self.labels_count = labels_count

        if embeddings:
            # id to vector mapping is provided as matrix
            self.embed = nn.Embedding.from_pretrained(embeddings)
        elif embeddings_path:
            # load id to vector mapping from file
            # Load from txt file (in word2vec format)
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)

            # Convert to PyTorch tensor
            weights = torch.FloatTensor(w2v_model.vectors)
            self.embed = nn.Embedding.from_pretrained(weights)
        else:
            raise ValueError('Either `embeddings` or `embeddings_path` must be set!')

        self.embedding_dim = self.embed.embedding_dim

        # Concat mode
        self.concat = concat
        self.concat_func, self.concat_dim = get_concat(concat, self.embedding_dim)

        self.mlp = get_mlp(input_dim=self.concat_dim,
                           output_dim=self.labels_count,
                           hidden_dim=mlp_dim,
                           hidden_layers_count=mlp_layers_count,
                           activation_cls=nn.ReLU)

        self.prob = self.get_classification_probability_layer(prob)

    def forward(self, input_ids_a, input_ids_b):
        a = self.embed(input_ids_a)
        b = self.embed(input_ids_b)

        concat_output = self.concat_func(a, b)

        output = self.mlp(concat_output)

        if self.prob:
            return self.prob(output)

        return output