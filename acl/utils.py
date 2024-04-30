import re
import logging

logger = logging.getLogger(__name__)


def get_sorted_pair(a, b):
    # ensure citation pair is always in same order
    if a > b:
        return (a, b)
    else:
        return (b, a)


def to_label(t, labels):
    if t in labels:
        return t
    else:
        return 'other'


def normalize_title(t):
    if t:
        t = t.replace('.', ' ').replace('-', ' ').strip().lower()
        #t = re.sub(r'\W+', '', t)
        return t


def normalize_section(title):
    if title:
        return re.sub(r'[\.0-9]', '',
                      title.
                      strip() \
                      .lower() \
                      .replace('conclusions', 'conclusion') \
                      .replace('methodology', 'method') \
                      .replace('methods', 'method') \
                      .replace('related works', 'related work') \
                      .replace('models', 'model') \
                      .replace('datasets', 'dataset') \
                      .replace('our ', '') \
                      .replace('evaluations', 'evaluation') \
                      .replace('experiments', 'experiment')
                      ).strip()
        # .replace('conclusion and future perspectives', 'conclusion')\
        # .replace('materials and methods', 'methods')


def get_text_from_doc(doc) -> str:
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

    return text


def get_text_from_doc_id(doc_id: str, doc_index) -> str:
    """

    Build document text from title + abstract

    :param doc_id: S2-id
    :param doc_index: S2-id to S2-paper data
    :return: Document text
    """

    if doc_id in doc_index:
        return get_text_from_doc(doc_index[doc_id])
    else:
        raise ValueError(f'Document not found in index: {doc_id}')


# resolve 'and' titles and filter for out-of-index docs
def resolve_and_sect_titles(items, doc_index=None):
    for from_s2_id, to_s2_id, sect_generic, sect_title, sect_marker in items:
        if doc_index and (from_s2_id not in doc_index or to_s2_id not in doc_index):
            # One of the IDs does not exist in document index
            continue

        sect_title = normalize_section(sect_title)

        if sect_title:
            # Resolve combined sections
            for t in sect_title.split(' and '):
                if t:
                    yield (from_s2_id, to_s2_id, t, sect_marker)
