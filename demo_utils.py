import requests
from transformers import AutoTokenizer

from models.bert import BertForMultiLabelSequenceClassification


def get_paper(doc_id):
    res = requests.get(f'https://api.semanticscholar.org/v1/paper/{doc_id}')

    if res.status_code == 200:
        return res.json()
    else:
        raise ValueError(f'Cannot load paper from S2 API: {doc_id}')


def get_prediction(model_name_or_path: str, from_id, to_id):
    from_doc = get_paper(from_id)
    to_doc = get_paper(to_id)

    if 'acl' in model_name_or_path:
        labels = ['introduction', 'related work', 'experiment', 'conclusion', 'results', 'background', 'discussion',
                  'evaluation', 'method', 'other', 'none']
    else:
        labels = ['discussion', 'introduction', 'conclusion', 'results', 'methods', 'background', 'materials', 'virus',
                  'future work', 'other', 'none']

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_name_or_path)

    model_input = tokenizer.batch_encode_plus(
        [(from_doc['title'] + '\n' + from_doc['abstract'], to_doc['title'] + '\n' + to_doc['abstract'])],
        pad_to_max_length=True, truncation_strategy='longest_first', return_token_type_ids=True,
        return_attention_masks=True, return_tensors='pt', max_length=512
    )

    model_out = model(**model_input)

    pred_scores = model_out[0].detach().numpy()[0]
    pred_labels = [label for idx, label in enumerate(labels) if pred_scores[idx] > 0.]

    return pred_scores, pred_labels, from_doc, to_doc
