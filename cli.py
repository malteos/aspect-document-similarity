import logging
import fire

from commands import word_vectors, compute_doc_vecs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    fire.Fire({
        'compute_doc_vecs': compute_doc_vecs.compute_doc_vecs,
        'extract_text': word_vectors.extract_text,
        'train_fasttext': word_vectors.train_fasttext,
    })
