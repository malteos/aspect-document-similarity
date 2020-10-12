import logging

import torch.nn as nn

from models.utils import get_concat, get_mlp

__all__ = [
    'ExperimentModel',
]

logger = logging.getLogger(__name__)


class ExperimentModel(nn.Module):
    labels_count = 1

    def forward(self, *input):
        raise NotImplementedError()

    def get_single_device(self):
        """
        If all parameters are on a single device, use this method to get current device.
        See: https://github.com/pytorch/pytorch/issues/7460
        """
        return next(self.parameters()).device
        
    def get_classification_probability_layer(self, mode='auto'):
        logger.debug(f'Classification probability layer with {mode}')
        if mode == 'auto':
            logger.debug(f'Auto-mode; labels count = {self.labels_count}')
            if self.labels_count == 1:
                return self.get_classification_probability_layer('sigmoid')
            else:
                return self.get_classification_probability_layer('softmax')
        elif mode == 'sigmoid':
            return nn.Sigmoid()
        elif mode == 'softmax':
            return nn.Softmax(dim=0)
        elif mode == 'none':
            return None
        else:
            raise ValueError('Unsupported mode')

