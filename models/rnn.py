from torch.nn import BCEWithLogitsLoss
from transformers import PreTrainedModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class RNNForMultiLabelSequenceClassification(nn.Module):
    """

    LSTM/GRU with GloVe/FastText word embeddings

    forward() compatible with Tranformers Trainer

    """

    def __init__(self, word_vectors, hidden_size=50, num_labels=2, num_layers=1, dropout=0., rnn='lstm'):
        super(RNNForMultiLabelSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.word_hidden_state = torch.zeros(2, 1, hidden_size)

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=word_vectors.shape[0], embedding_dim=word_vectors.shape[1])

        if rnn == 'gru':
            self.rnn = nn.GRU(
                input_size=word_vectors.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(
                input_size=word_vectors.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError('Unknown RNN type')

        self._create_weights(mean=0.0, std=0.05)

        self.word_attention = nn.Linear(2 * hidden_size, 50)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(50, 1, bias=False)

        self.classifier = nn.Linear(2 * hidden_size, self.num_labels)

        # torch.set_printoptions(threshold=10000)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        word_ids_lengths = attention_mask.sum(axis=1)
        word_embeddings = self.lookup(input_ids)

        packed_word_embeddings = pack_padded_sequence(word_embeddings,
                                                      lengths=word_ids_lengths,
                                                      batch_first=True,
                                                      enforce_sorted=False)

        words_representation, _ = self.rnn(packed_word_embeddings)
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        word_attention = self.word_attention(words_representation.data)
        word_attention = torch.tanh(word_attention)

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        word_attention = self.word_context_vector(word_attention).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = word_attention.max()  # scalar, for numerical stability during exponent calculation
        word_attention = torch.exp(word_attention - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        word_attention, _ = pad_packed_sequence(PackedSequence(data=word_attention,
                                                               batch_sizes=words_representation.batch_sizes,
                                                               sorted_indices=words_representation.sorted_indices,
                                                               unsorted_indices=words_representation.unsorted_indices),
                                                batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = word_attention / torch.sum(word_attention, dim=1,
                                                 keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(words_representation,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # gets the representation for the sentence
        sentences = sentences.sum(dim=1)  # (n_sentences)

        logits = self.classifier(sentences)

        outputs = (logits, sentences)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            outputs = (loss,) + outputs

        return outputs
