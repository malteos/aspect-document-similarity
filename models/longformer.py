from torch.nn import BCEWithLogitsLoss
from transformers import RobertaForSequenceClassification, BertPreTrainedModel, LongformerConfig, LongformerModel, \
    LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.modeling_roberta import RobertaClassificationHead


class LongformerForMultiLabelSequenceClassification(BertPreTrainedModel):
    config_class = LongformerConfig
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = LongformerModel(config)
        self.classifier = RobertaClassificationHead(config)

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
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            # Single-label classification (as in RobertaForSequenceClassification)
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)