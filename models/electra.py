import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification, ElectraForSequenceClassification


class ElectraForMultiLabelSequenceClassification(ElectraForSequenceClassification):
    """Electra model for classification.
    This module is composed of Electra BERT model with a linear layer on top of
    the pooled output.
    """

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
        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + discriminator_hidden_states[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Single-label classification (as in BertForSequenceClassification)
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
