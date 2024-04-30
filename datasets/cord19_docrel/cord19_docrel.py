from __future__ import absolute_import, division, print_function

import json
import os

import nlp
from pyarrow import csv

_DESCRIPTION = """Aspect-oriented Document Similarity from the CORD-19 dataset"""

_HOMEPAGE = "https://github.com/malteos/aspect-document-similarity"

_CITATION = """
@InProceedings{Ostendorff2020b,
  title = {Aspect-based Document Similarity for Research Papers},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020},
  author = {Ostendorff, Malte and Ruas, Terry and Blume, Till and Gipp, Bela and Rehm, Georg},
  year = {2020},
  month = {Dec.},
}
"""

LABEL_CLASSES = ['discussion',
                 'introduction',
                 'conclusion',
                 'results',
                 'methods',
                 'background',
                 'materials',
                 'virus',
                 'future work',
                 'other',
                 'none']

DATA_URL = "http://datasets.fiq.de/cord19_docrel.tar.gz"

DOC_A_COL = "from_doi"

DOC_B_COL = "to_doi"

LABEL_COL = "label"


def get_train_split(k):
    return nlp.Split(f'fold_{k}_train')


def get_test_split(k):
    return nlp.Split(f'fold_{k}_test')


class Cord19DocrelConfig(nlp.BuilderConfig):
    def __init__(self, features, data_url, **kwargs):
        super(Cord19DocrelConfig, self).__init__(version=nlp.Version("0.1.0"), **kwargs)
        self.features = features
        self.data_url = data_url


class Cord19Docrel(nlp.GeneratorBasedBuilder):
    """CORD-19 document relation dataset."""

    BUILDER_CONFIGS = [
        Cord19DocrelConfig(
            name="docs",
            description="document text and meta data",
            features={
                "doi": nlp.Value("string"),
                "cord19_id": nlp.Value("string"),
                "s2_id": nlp.Value("string"),
                "title": nlp.Value("string"),
                "abstract": nlp.Value("string"),
                "arxivId": nlp.Value("string"),
                "venue": nlp.Value("string"),
                "year": nlp.Value("int16"),
                "citations_count": nlp.Value("int32"),
                "references_count": nlp.Value("int32"),
                "authors": nlp.Sequence(nlp.Value('string', id='author_name')),
            },
            data_url=DATA_URL,
        ),
        Cord19DocrelConfig(
            name="relations",
            description=" relation data",
            features={
                DOC_A_COL: nlp.Value("string"),
                DOC_B_COL: nlp.Value("string"),
                LABEL_COL: nlp.Sequence(nlp.Value('string', id='label'))
            },
            data_url=DATA_URL,
        ),
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION + self.config.description,
            features=nlp.Features(self.config.features),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        arch_path = dl_manager.download_and_extract(self.config.data_url)

        if "relations" in self.config.name:
            train_file = "train.csv"
            test_file = "test.csv"

            generators = []

            for k in [1, 2, 3, 4]:
                folds_path = os.path.join(arch_path, 'folds', str(k))
                generators += [
                    nlp.SplitGenerator(
                        name=get_train_split(k),
                        gen_kwargs={'filepath': os.path.join(folds_path, train_file)}
                    ),
                    nlp.SplitGenerator(
                        name=get_test_split(k),
                        gen_kwargs={'filepath': os.path.join(folds_path, test_file)}
                    )
                ]
            return generators

        elif "docs" in self.config.name:
            # docs
            docs_file = os.path.join(arch_path, "docs.jsonl")

            return [
                nlp.SplitGenerator(name=nlp.Split('docs'), gen_kwargs={"filepath": docs_file}),
            ]
        else:
            raise ValueError()

    @staticmethod
    def get_dict_value(d, key, default=None):
        if key in d:
            return d[key]
        else:
            return default

    def _generate_examples(self, filepath):
        """Generate docs + rel examples."""

        if "relations" in self.config.name:
            df = csv.read_csv(filepath).to_pandas()

            for idx, row in df.iterrows():
                yield idx, {
                    DOC_A_COL: row[DOC_A_COL],
                    DOC_B_COL: row[DOC_B_COL],
                    LABEL_COL: row[LABEL_COL].split(','),
                }

        elif self.config.name == "docs":

            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    doc = json.loads(line)

                    yield i, {
                        'doi': str(self.get_dict_value(doc, 'doi')),  # cast to str otherwise float
                        'cord19_id': self.get_dict_value(doc, 'cord19_id'),
                        's2_id': self.get_dict_value(doc, 's2_id'),
                        'title': self.get_dict_value(doc, 'title'),
                        'abstract': self.get_dict_value(doc, 'abstract'),
                        'arxivId': self.get_dict_value(doc, 'arxivId'),
                        'venue': str(self.get_dict_value(doc, 'venue') or ''),
                        'year': int(self.get_dict_value(doc, 'year', 0) or 0),
                        'citations_count': int(self.get_dict_value(doc, 'citations_count', 0) or 0),
                        'references_count': int(self.get_dict_value(doc, 'references_count', 0) or 0),
                        'authors': self.get_dict_value(doc, 'authors', []),
                    }

