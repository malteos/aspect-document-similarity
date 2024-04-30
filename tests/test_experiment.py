from unittest import TestCase

from torch.nn import BCELoss
from tqdm import tqdm

from experiments import Experiment
from experiments.utils import flatten
from models.transformers import JointBERT
from wiki.data_helpers import JointBERTWikiDataHelper


class ExperimentTest(TestCase):
    def test_cls_init(self):

        exp = Experiment(
            # random_seed=0,
            epochs=1,
            model_cls='models.JointBERT',
            model_params={
                'bert_model_path': '/Volumes/data/repo/data/bert/bert-base-cased',
                'labels_count': 3,
            },
            loss_func_cls='torch.nn.BCELoss',  # loss,
            model_output_to_loss_input=lambda ys: ys.double(),
            data_helper_cls='wiki.data_helpers.JointBERTDataHelper',
            data_helper_params={
                'wiki_relations_path': '../wiki/relations.csv',
                'wiki_articles_path': '../wiki/docs.pickle',
                'labels': ['employer', 'country_of_citizenship'],
                # 'employer' # 'capital' # 'country_of_citizenship' #'educated_at' # 'opposite_of'
                'label_col': 'relation_name',
                'negative_sampling_ratio': 1.,
                'train_test_split': 0.7,
                'max_seq_length': 512,
                'train_batch_size': 4,
                'test_batch_size': 4,
                'bert_model_path': '/Volumes/data/repo/data/bert/bert-base-cased',
                # 'bert_tokenizer_cls': '',
                'bert_tokenizer_params': {
                    'do_lower_case': False,
                },
                'df_limit': 3,
            },
            tqdm_cls='tqdm.tqdm',
            output_dir='../output',
        )

        assert isinstance(exp.model, JointBERT)
        assert isinstance(exp.data_helper, JointBERTWikiDataHelper)
        assert isinstance(exp.loss_func, BCELoss)
        assert tqdm == exp.tqdm_cls

        print(flatten(exp.to_dict()))

        exp.run()