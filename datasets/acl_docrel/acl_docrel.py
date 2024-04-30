from __future__ import absolute_import, division, print_function

import json
import os

import nlp
from pyarrow import csv

_DESCRIPTION = """Aspect-oriented Document Similarity from the ACL-Anthology dataset"""

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

LABEL_CLASSES = ['introduction',
                 'related work',
                 'experiment',
                 'conclusion',
                 'results',
                 'background',
                 'discussion',
                 'evaluation',
                 'method',
                 #'previous work',
                 'other',
                 'none']

DATA_URL = "http://datasets.fiq.de/acl_docrel.tar.gz"

def get_train_split(k):
    return nlp.Split(f'fold_{k}_train')


def get_test_split(k):
    return nlp.Split(f'fold_{k}_test')


class AclDocrelConfig(nlp.BuilderConfig):
    def __init__(self, features, data_url, **kwargs):
        super(AclDocrelConfig, self).__init__(version=nlp.Version("0.1.0"), **kwargs)
        self.features = features
        self.data_url = data_url


class AclDocrel(nlp.GeneratorBasedBuilder):
    """ACL anthology document relation dataset."""

    BUILDER_CONFIGS = [
        AclDocrelConfig(
            name="docs",
            description="document text and meta data",
            features={
                "s2_id": nlp.Value("string"),
                "title": nlp.Value("string"),
                "abstract": nlp.Value("string"),
                "arxivId": nlp.Value("string"),
                "doi": nlp.Value("string"),
                "venue": nlp.Value("string"),
                "year": nlp.Value("int16"),
                "citations_count": nlp.Value("int32"),
                "references_count": nlp.Value("int32"),
                "authors": nlp.Sequence(nlp.Value('string', id='author_name')),
            },
            data_url=DATA_URL,
        ),
        AclDocrelConfig(
            name="relations",
            description=" relation data",
            features={
                "from_s2_id": nlp.Value("string"),
                "to_s2_id": nlp.Value("string"),
                "label": nlp.Sequence(nlp.Value('string', id='label'))
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

        if self.config.name == "relations":
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

        elif self.config.name == "docs":
            # docs
            docs_file = os.path.join(arch_path, "docs.jsonl")

            return [
                nlp.SplitGenerator(name=nlp.Split('docs'), gen_kwargs={"filepath": docs_file}),
            ]
        else:
            raise ValueError()

    @staticmethod
    def get_s2_value(s2, key, default=None):
        if key in s2:
            return s2[key]
        else:
            return default

    def _generate_examples(self, filepath):
        """Generate docs + rel examples."""

        if self.config.name == "relations":
            df = csv.read_csv(filepath).to_pandas()

            for idx, row in df.iterrows():
                yield idx, dict(from_s2_id=row['from_s2_id'], to_s2_id=row['to_s2_id'], label=row['label'].split(','))

        elif self.config.name == "docs":

            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    s2 = json.loads(line)

                    yield i, {
                        's2_id': self.get_s2_value(s2, 'paperId'),
                        'title': self.get_s2_value(s2, 'title'),
                        'abstract': self.get_s2_value(s2, 'abstract'),
                        'doi': self.get_s2_value(s2, 'doi'),
                        'arxivId': self.get_s2_value(s2, 'arxivId'),
                        'venue': self.get_s2_value(s2, 'venue'),
                        'year': self.get_s2_value(s2, 'year', 0),
                        'citations_count': len(self.get_s2_value(s2, 'citations', [])),
                        'references_count': len(self.get_s2_value(s2, 'references', [])),
                        'authors': [a['name'] for a in self.get_s2_value(s2, 'authors', []) if 'name' in a],
                    }
