import torch
from torch.nn import BCEWithLogitsLoss
from transformers import BartForSequenceClassification


class BartForMultiLabelSequenceClassification(BartForSequenceClassification):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output,
            # Single label
            # loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1, self.config.num_labels))

            outputs = (loss,) + outputs

        return outputs

