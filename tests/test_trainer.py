import os
from unittest import TestCase

import pandas as pd
from nlp import load_dataset

from acl.trainer_utils import get_label_classes_from_nlp_dataset
from datasets.acl_docrel.acl_docrel import get_train_split, get_test_split
from experiments.environment import get_env


class TrainerTest(TestCase):
    def __init__(self, *args, **kwargs):
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_WATCH"] = "false"

        super().__init__(*args, **kwargs)
        self.env = get_env()

    def test_label_classes(self):
        ds = "./datasets/acl_docrel/acl_docrel.py"

        ls = get_label_classes_from_nlp_dataset(ds)

        self.assertEqual(['introduction',
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
                 'none'], ls)

    def test_load_dataset(self):
        pass

    def test_load_datasets_and_compare_label_class_distribution(self):
        cache_dir = '../data/nlp_cache'
        acl_ds = "../datasets/acl_docrel/acl_docrel.py"
        cv_fold = 1

        train_ds = load_dataset(acl_ds,
                                name='relations',
                                cache_dir=cache_dir,
                                split=get_train_split(cv_fold))
        test_ds = load_dataset(acl_ds,
                               name='relations',
                               cache_dir=cache_dir,
                               split=get_test_split(cv_fold))

        labels = [l for r in test_ds for l in r['label']] + [l for r in train_ds for l in r['label']]
        df = pd.DataFrame(labels, columns=['label'])

        print('ACL')
        print(df['label'].value_counts())

        print('Pairs: %s '(len(train_ds) + len(test_ds)))


        ######

        cord19_ds = "../datasets/cord19_docrel/cord19_docrel.py"
        train_ds = load_dataset(cord19_ds,
                                name='relations',
                                cache_dir=cache_dir,
                                split=get_train_split(cv_fold))
        test_ds = load_dataset(cord19_ds,
                               name='relations',
                               cache_dir=cache_dir,
                               split=get_test_split(cv_fold))

        labels = [l for r in test_ds for l in r['label']] + [l for r in train_ds for l in r['label']]
        df = pd.DataFrame(labels, columns=['label'])

        print('CORD19')
        print(df['label'].value_counts())

        print('Pairs: %s ' (len(train_ds) + len(test_ds)))


    def test_dataset_splits(self):
        cache_dir = '../data/nlp_cache'

        for ds in ["../datasets/acl_docrel/acl_docrel.py", "../datasets/cord19_docrel/cord19_docrel.py"]:
            print(ds)

            train_count = 0
            test_count = 0

            for cv_fold in [1,2,3,4]:
                train_ds = load_dataset(ds,
                                        name='relations',
                                        cache_dir=cache_dir,
                                        split=get_train_split(cv_fold))

                train_count += len(train_ds)

                test_ds = load_dataset(ds,
                                       name='relations',
                                       cache_dir=cache_dir,
                                       split=get_test_split(cv_fold))
                test_count += len(test_ds)

            print('Train: %s' % (train_count / 4))
            print('Test: %s' % (test_count / 4))
            print()