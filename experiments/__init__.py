import logging
import os
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from experiments.data_helpers import DataHelper
from experiments.utils import get_linear_schedule_with_warmup
from models import ExperimentModel

logger = logging.getLogger(__name__)


class Experiment(object):
    """

    """
    name = None  # type: str
    random_seed = None
    model = None  # type: ExperimentModel
    model_cls = None
    model_params = None
    train_data_loader = None
    test_data_loader = None
    device = None
    optimizer_cls = None
    optimizer_params = None  # type: dict
    loss_func = None
    loss_func_cls = None
    weight_decay = 0.0

    tqdm_cls = None  #  use tqdm or tqdm_notebook
    data_helper = None  # type: DataHelper
    data_helper_cls = None
    data_helper_params = None  # type: dict
    output_dir = None
    epochs = None  # type: int
    labels = None
    test_every_n_epoch = 0
    test_on_train = False  # for debugging

    classification_threshold = None
    classification_by_max = True
    tensorboard_writer = None  # type: SummaryWriter
    tensorboard_params = None
    
    # training and evaluation log
    _config = {}
    reports = {}
    outputs = {}
    train_loss = {}

    logged_parameter = {  # TODO define hyper parameter that are included in log file
    }
    
    def __init__(self, **kwargs):
        to_class_name = ['model_cls', 'optimizer_cls', 'data_helper_cls', 'tqdm_cls', 'loss_func_cls']
        exclude_from_config = [
            'model_output_to_loss_input',
            'loss_func',
            'data_loader_to_loss_input',
        ]
        self._config = {}

        # set class attributes
        for k, v in kwargs.items():

            # store config
            if k in exclude_from_config:
                pass
            elif k in to_class_name and not isinstance(v, str):
                self._config[k] = v.__class__.__name__
            else:
                self._config[k] = v

            # dynamic import for classes
            if k in to_class_name and isinstance(v, str):
                package, module_name = v.rsplit('.', 1)
                module = import_module(package)
                v = getattr(module, module_name)  # replace value with actual class

            # set class attributes
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f'Unknown attribute: {k}')

        # actual init
        if not self.model and self.model_cls:
            self.model = self.model_cls(**self.model_params)

        if not self.data_helper and self.data_helper_cls:
            self.data_helper = self.data_helper_cls(**self.data_helper_params)

        if self.tqdm_cls:
            self.data_helper.tqdm_cls = self.tqdm_cls

        if self.loss_func_cls and not self.loss_func:
            self.loss_func = self.loss_func_cls()

        if self.tensorboard_params and 'auto' in self.tensorboard_params and 'dir' in self.tensorboard_params and self.tensorboard_params['auto']:
            self.tensorboard_writer = SummaryWriter(os.path.join(self.tensorboard_params['dir'], self.get_name()))
            # self.tensorboard_writer.add_hparam  # TODO is not available in Pytorch 1.1.0
            
            logger.info('Tensorboard enabled')

    def get_name(self):
        if self.name is None:
            model_name = self.model.__class__.__name__.replace('Model', '')
            data_helper_name = self.data_helper.__class__.__name__.replace('DataHelper', '')
            
            now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            
            self.name = f'{model_name}_{data_helper_name}_{now}'
            
            logger.info(f'Model name: {self.name}')
            
        return self.name
    
    def prepare(self, cuda_device):
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        if not torch.cuda.is_available():
            logger.error('CUDA GPU is not available')
            exit(1)

    @staticmethod
    def data_loader_to_loss_input(ys):
        """
        Converts data loader output to loss function input

        :param ys:
        :return:
        """
        return ys

    @staticmethod
    def model_output_to_loss_input(ys):
        """
        Converts model output to loss function input

        :param ys:
        :return:
        """
        return ys

    @staticmethod
    def convert_one_hot_to_label_indexes(ys):
        """

        Use this when using CrossEntropyLoss as loss function (does not use actual OneHotEncoder from data helper)

        :param ys: one hot encoded labels (batch size x labels count)
        :param dim:
        :return: label indexes (batch size x 1)
        """

        return torch.argmax(ys, 1) #torch.nonzero(ys)[dim]

    @staticmethod
    def convert_to_double(tensor):
        return tensor.double()

    @staticmethod
    def get_best_thresholds(labels, correct_output, predictions, plot=False, average='micro', none_label='none'):
        """
        Hyper parameter search for best classification threshold
        """
        
        if len(labels) == 2:
            # binary
            logger.info('Finding threshold for binary classification')
            
            f_max = 0.
            t_max = 0.5

            if len(correct_output.shape) > 1:
                # one-hot encoding
                correct_output = correct_output[:, 0]
                predictions = predictions[:, 0]

            for t in np.linspace(0.1, 0.99, num=50):
                p, r, f, _ = precision_recall_fscore_support(correct_output, np.where(predictions > t, 1, 0),
                                                             average='micro')

                if f > f_max:
                    f_max = f
                    t_max = t
            return t_max, f_max
        
        elif len(labels) > 2:
            # multi label
            logger.info('Finding threshold for multi-label classification')
            
            t_max = [0.5] * len(labels)
            f_max = [0] * len(labels)

            for i, label in enumerate(labels):
                ts = []
                fs = []

                # exclude none label
                if label == none_label:
                    continue

                for t in np.linspace(0.1, 0.99, num=50):
                    p, r, f, _ = precision_recall_fscore_support(correct_output[:, i], np.where(predictions[:, i] > t, 1, 0),
                                                                 average=average)
                    ts.append(t)
                    fs.append(f)

                    if f > f_max[i]:
                        f_max[i] = f
                        t_max[i] = t

                if plot:
                    raise NotImplementedError()
                    # print(f'LABEL: {label}')
                    # print(f'f_max: {f_max[i]}')
                    # print(f't_max: {t_max[i]}')
                    #
                    # plt.scatter(ts, fs)
                    # plt.show()

            return t_max, f_max
        else:
            raise ValueError(f'Invalid number of labels provided: {labels}')

    def progress(self, iterator, **kwargs):
        if self.tqdm_cls:
            # Override iterator to display progress
            iterator = self.tqdm_cls(iterator, **kwargs)
        return iterator

    def model_output_to_prediction(self, ys_pred):
        return ys_pred

    def get_test_output(self, all_ys, all_ys_pred):

        if hasattr(self.data_helper, 'label_encoder'):
            labels = self.data_helper.label_encoder.classes_
        else:
            labels = self.data_helper.labels
        
        logger.info(f'Classification labels: {labels}')

        # TODO maybe check: classification_threshold is None
        if self.classification_by_max:
            # max probablity is the prediction ---> multi-class, single-label
            correct_outputs = all_ys

            predicted_outputs = np.zeros_like(correct_outputs)
            predicted_outputs[np.arange(len(all_ys_pred)), all_ys_pred.argmax(1)] = 1

            correct_outputs = correct_outputs.argmax(1)
            predicted_outputs = predicted_outputs.argmax(1)

            report = classification_report(correct_outputs,
                                           predicted_outputs,
                                           labels=range(len(labels)),
                                           target_names=labels,
                                           output_dict=True)

        else:
            # multi-class, multi-label
            if not self.classification_threshold:
                threshold, _ = self.get_best_thresholds(labels, all_ys, all_ys_pred)
            else:
                threshold = self.classification_threshold

            logger.info(f'Threshold: {threshold}')

            if isinstance(threshold, np.float64):
                # binary
                if len(all_ys_pred.shape) > 1:
                    # one-hot encoding
                    predicted_outputs = np.where(all_ys_pred[:, 0] > threshold, 1, 0)
                    correct_outputs = all_ys[:, 0]
                else:
                    predicted_outputs = np.where(all_ys_pred > threshold, 1, 0)
                    correct_outputs = all_ys

            else:
                # multi label
                predicted_outputs = np.where(all_ys_pred > threshold, 1, 0)
                correct_outputs = all_ys

            report = classification_report(correct_outputs, predicted_outputs, target_names=labels, output_dict=True)
        # cm = confusion_matrix(correct_outputs, predicted_outputs)

        return report, correct_outputs, predicted_outputs

    def test_old(self, data_loader):
        if self.test_on_train:
            logger.warning('TESTING ON TRAIN DATA (USE THIS ONLY FOR DEBUGGING!)')

        self.model.eval()

        output_ids = []

        all_ys_pred = None
        all_ys = None
        batch_iterator = data_loader

        with torch.no_grad():  # Disable gradient for evaluation
            for batch_data in self.progress(batch_iterator, desc=f'Evaluate'):
                # in case GPU is enabled, this sends data to GPU
                batch_data = tuple(t.to(self.device) for t in batch_data)

                xs = batch_iterator.get_x_from_batch(batch_data)
                ys = batch_iterator.get_y_from_batch(batch_data)

                ys_pred = self.model(*xs)

                # From GPU back to CPU
                ys_pred = self.model_output_to_prediction(ys_pred).cpu().detach().numpy()
                ys = ys.cpu().detach().numpy()

                # all_ys_pred = ys_pred if all_ys_pred is None else np.vstack((all_ys_pred, ys_pred))
                # all_ys = ys if all_ys is None else np.vstack((ys, all_ys))
                all_ys_pred = ys_pred if all_ys_pred is None else np.concatenate((all_ys_pred, ys_pred))
                all_ys = ys if all_ys is None else np.concatenate((ys, all_ys))
        
        report, correct_outputs, predicted_outputs = self.get_test_output(all_ys, all_ys_pred)

        return report, (all_ys, all_ys_pred, correct_outputs, predicted_outputs)

    def test(self, data_loader, batch_limit=0):
        self.model.eval()

        correct_out = []
        predicted_out = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.progress(data_loader)):
                batch_data = tuple(t.to(self.device) for t in batch_data)
                xs = data_loader.get_x_from_batch(batch_data)
                ys = data_loader.get_y_from_batch(batch_data)

                # Pass to model
                ys_pred = self.model(*xs)

                correct_out += ys.cpu().numpy().tolist()
                predicted_out += self.model_output_to_prediction(ys_pred).cpu().detach().numpy().tolist()

                if 0 < batch_limit <= batch_idx:
                    logger.info(f'Stop with batch limit = {batch_limit}')
                    break

        correct_out = np.array(correct_out)
        predicted_out = np.array(predicted_out)

        report, correct_outputs, predicted_outputs = self.get_test_output(correct_out, predicted_out)
        return report, (correct_out, predicted_out, correct_outputs, predicted_outputs)

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self, mode=0):
        # Set GPU
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        self.set_device()
        
        logger.info(f'Running on device: {self.device}')

        # Data loaders
        if not self.train_data_loader and not self.test_data_loader:
            # self.data_helper = self.data_helper_cls(**self.data_helper_params)
            self.train_data_loader, self.test_data_loader = self.data_helper.get_data_loaders()

            if self.test_on_train:
                logger.warning('TESTING ON TRAIN DATA (USE THIS ONLY FOR DEBUGGING!)')
                self.test_data_loader = self.train_data_loader

        self.reports = {}
        self.outputs = {}

        if self.device.type == 'cuda':
            self.model = self.model.cuda()
            logger.info('Model loaded to GPU')

        self.train()

        return self.reports

    def train(self, start_epoch=1):

        logger.info('New training')

        LEARNING_RATE = 2e-5  # 2e-6 does not work (?)

        optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)

        loss_func = BCELoss()

        for epoch_num in range(start_epoch, start_epoch + self.epochs):
            self.model.train()
            train_loss = 0

            # print(f'Epoch: {epoch_num}/{start_epoch + self.epochs}')

            # for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            for batch_idx, batch_data in enumerate(self.progress(self.train_data_loader, desc=f'Training Epoch: {epoch_num}/{self.epochs}')):
                batch_data = tuple(t.to(self.device) for t in batch_data)
                xs = self.train_data_loader.get_x_from_batch(batch_data)
                ys = self.train_data_loader.get_y_from_batch(batch_data)

                # Pass to model
                ys_pred = self.model(*xs)

                batch_loss = loss_func(
                    self.model_output_to_loss_input(ys_pred),
                    self.data_loader_to_loss_input(ys),
                )

                train_loss += batch_loss.item()

                self.model.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # clear_output(wait=True)

            # print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(self.train_data_loader) / self.data_helper.train_batch_size,
            #                                          train_loss / (step_num + 1)))

            # Loss
            train_loss = train_loss / (batch_idx + 1)
            logger.info(f'Loss/Train after {epoch_num} epochs: {train_loss}')

            # test if requested or last epoch
            if epoch_num == self.epochs or (
                    self.test_every_n_epoch > 0 and (epoch_num % self.test_every_n_epoch) == 0):
                report, outputs = self.test(self.test_data_loader)

                self.reports[epoch_num] = report
                self.outputs[epoch_num] = outputs

                for avg in ['micro avg', 'macro avg']:
                    if avg in report:
                        for metric in ['precision', 'recall', 'f1-score']:
                            if metric in report[avg]:
                                avg_label = avg.replace(' ', '_')

                                logger.info(f'{avg_label}-{metric}/Test: {report[avg][metric]}')

                                if self.tensorboard_writer:
                                    self.tensorboard_writer.add_scalar(f'{avg_label}-{metric}/Test',
                                                                       report[avg][metric], epoch_num)


            logger.info(str(torch.cuda.memory_allocated(self.device) / 1000000) + 'M')

        logger.info('Debug training completed')

        return self.reports

    def train_old(self, start_epoch=1):
        raise DeprecationWarning('Use train')

    #     ####
    #
    #     #learning_rate = 5e-5  # The initial learning rate for Adam.
    #     #adam_epsilon = 1e-8  # Epsilon for Adam optimizer.
    #     #warmup_steps = 0  # Linear warmup over warmup_steps.
    #     #t_total =  len(self.train_data_loader) // 1 * self.epochs
    #     # Prepare optimizer and schedule (linear warmup and decay)
    #     #no_decay = ["bias", "LayerNorm.weight"]
    #     #optimizer_grouped_parameters = [
    #     #    {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
    #     #     "weight_decay": self.weight_decay},
    #     #    {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    #     #]
    #     #optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    #     #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #     #                                            num_training_steps=t_total)
    #
    #     ########
    #
    #     logger.info('Standard training')
    #
    #     # Training
    #     scheduler = None
    #     optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
    #
    #     self.model.zero_grad()
    #
    #     # actual training loop
    #     for epoch_num in range(start_epoch, start_epoch + self.epochs):
    #         train_loss = 0
    #
    #         # iterate over each batch
    #         for batch_idx, batch_data in enumerate(self.progress(self.train_data_loader, desc=f'Training Epoch: {epoch_num}/{self.epochs}')):
    #             # switch model to training mode, clear gradient accumulators
    #             self.model.train()
    #             # optimizer.zero_grad()
    #
    #             # in case GPU is enabled, this sends data to GPU
    #             batch_data = tuple(t.to(self.device) for t in batch_data)
    #
    #             xs = self.train_data_loader.get_x_from_batch(batch_data)
    #             ys = self.train_data_loader.get_y_from_batch(batch_data)
    #
    #             # Pass to model
    #             ys_pred = self.model(*xs)
    #
    #             batch_loss = self.loss_func(
    #                 self.model_output_to_loss_input(ys_pred),
    #                 self.data_loader_to_loss_input(ys),
    #             )
    #
    #             batch_loss.backward()
    #             train_loss += batch_loss.item()
    #
    #             if scheduler:
    #                 scheduler.step()  # Update learning rate schedule
    #
    #             optimizer.step()
    #             self.model.zero_grad()
    #
    #         # test if requested or last epoch
    #         if epoch_num == self.epochs or (self.test_every_n_epoch > 0 and (epoch_num % self.test_every_n_epoch) == 0):
    #             report, outputs = self.test(self.test_data_loader)
    #
    #             self.reports[epoch_num] = report
    #             self.outputs[epoch_num] = outputs
    #
    #
    #             for avg in ['micro avg', 'macro avg']:
    #                 if avg in report:
    #                     for metric in ['precision', 'recall', 'f1-score']:
    #                         if metric in report[avg]:
    #                             avg_label = avg.replace(' ', '_')
    #
    #                             logger.info(f'{avg_label}-{metric}/Test: {report[avg][metric]}')
    #
    #                             if self.tensorboard_writer:
    #                                 self.tensorboard_writer.add_scalar(f'{avg_label}-{metric}/Test', report[avg][metric], epoch_num)
    #
    #         # Loss
    #         train_loss = train_loss / (batch_idx + 1)
    #         logger.info(f'Loss/Train after {epoch_num} epochs: {train_loss}')
    #
    #         self.train_loss[epoch_num] = train_loss
    #
    #         if self.tensorboard_writer:
    #             self.tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch_num)
    #
    #
    #     return self.reports
    
    def get_output_dir(self):
        if not self.output_dir:
            self.output_dir = os.path.join('./output', self.get_name())
            logger.debug(f'Output dir: {self.output_dir}')
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                logger.debug('Output dir created')
            
        return self.output_dir
        
    def load(self):
        logger.debug(f'Loading model from: {self.get_output_dir()}')
        self.model.load_state_dict(torch.load(os.path.join(self.get_output_dir(), 'model_weights')))
        pass
    
    def save(self):
        logger.debug(f'Saving model to: {self.get_output_dir()}')
        torch.save(self.model.state_dict(), os.path.join(self.get_output_dir(), 'model_weights'))
        # json.dump(self.config, f)

    def to_dict(self):
        return self._config

##


#
# exp = Experiment(
#     random_seed=0,
#     model_cls=JointBERT,
#     model_params={
#         'max_seq_length': 512,
#         'bert_model_path': '/Volumes/data/repo/data/bert/bert-base-cased',
#     },
#     optimizer_cls=Adam,
#     optimizer_params={
#         'lr': 2e-5,
#     },
#     epochs=5,
#     test_every_n_epoch=1,
#     loss_func=nn.BCELoss(),
#     model_output_to_loss_input=lambda ys: ys.float(),
#     data_helper_cls=JointBERTDataHelper,
#     data_helper_params={
#         'wiki_relations_path': '../wiki/relations.csv',
#         'wiki_articles_path': '../wiki/docs.pickle',
#         'labels': ['employer', 'capital'], #  'employer' # 'capital' # 'country_of_citizenship' #'educated_at' # 'opposite_of'
#         'label_col': 'relation_name',
#         'negative_samples_count': 1,
#         'train_test_split': 0.7,
#         'max_seq_length': 512,
#         'train_batch_size': 4,
#         'test_batch_size': 4,
#         'bert_model_path': '/Volumes/data/repo/data/bert/bert-base-cased',
#         # 'bert_tokenizer_cls': '',
#         'bert_tokenizer_params': {
#             'do_lower_case': False,
#         }
#     },
#     output_path='../output',
# )
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
#
# # exp.run()
#
# tensorboard_writer = SummaryWriter('../runs/experiment12')
#
# tensorboard_writer.add_scalar('Loss/Train', 40343.3, 1)
#

#self.tensorboard_writer.add_scalar('Macro-F1/Test', macro_f1, self.epoch)