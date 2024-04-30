import torch
import torch.nn as nn

import logging


logger = logging.getLogger(__name__)


def get_concat(concat: str, embedding_dim: int):
    """

    :param concat: Concatenation style
    :param embedding_dim: Size of inputs that are subject to concatenation
    :return: Function that performs concatenation, Size of concatenation output
    """
    concat_func = None
    concat_dim = None

    if concat == 'simple':
        concat_func = lambda a, b: torch.cat((a, b), dim=1)
        concat_dim = 2 * embedding_dim
    elif concat == 'dif':
        # x = np.abs(a-b)
        concat_func = lambda a, b: (a - b).abs()
        concat_dim = 1 * embedding_dim
    elif concat == 'prod':
        # x = a * b
        concat_func = lambda a, b: a * b
        concat_dim = 1 * embedding_dim
    elif concat == 'dif-prod':
        # x = np.hstack((np.abs(a-b), a * b))
        concat_func = lambda a, b: torch.cat(((a - b).abs(), a * b), dim=1)
        concat_dim = 2 * embedding_dim

    elif concat == '3d-prod':
        # x = np.hstack((a, b, a*b))
        concat_func = lambda a, b: torch.cat((a, b, a * b), dim=1)
        concat_dim = 3 * embedding_dim

    elif concat == '3d-dif':
        # x = np.hstack((a, b, np.abs(a-b)))
        concat_func = lambda a, b: torch.cat((a, b, (a - b).abs()), dim=1)
        concat_dim = 3 * embedding_dim
    elif concat == '4d-prod-dif':
        # x = np.hstack((a, b, a*b, np.abs(a-b)))
        concat_func = lambda a, b: torch.cat((a, b, a * b, (a - b).abs()), dim=1)
        concat_dim = 4 * embedding_dim

    else:
        raise ValueError('Unsupported concat mode')

    logger.debug(f'concat_dim = {concat_dim} ({concat})')
        
    return concat_func, concat_dim


def get_mlp(input_dim, output_dim, hidden_dim, hidden_layers_count=1, dropout_p=0., activation_cls=nn.ReLU):
    """
    Generate a fully-connected layer (MLP) with dynamic input, output and hidden dimension, and hidden layer count.
    
    - when dropout_p > 0, then dropout is applied with given probability after the activation function.
    
    :param input_dim:  
    :return: Sequential layer
    """
    layers = [
        # first layer
        nn.Linear(input_dim, hidden_dim),
        activation_cls(),
    ]

    if dropout_p > 0:
        layers.append(nn.Dropout(dropout_p))

    for layer_idx in range(1, hidden_layers_count):
        layers.append(nn.Linear(hidden_dim, hidden_dim)),
        layers.append(activation_cls()),

        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

    # last layer
    layers.append(nn.Linear(hidden_dim, output_dim))

    # TODO fill linear layers
    # nn.init.xavier_normal_(self.classifier.weight)
    # Fills the input Tensor with values according to the method described in “Understanding the difficulty of training deep feedforward neural networks” - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
    # kaiming_normal_
    # Fills the input Tensor with values according to the method described in “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015), using a normal distribution.

    return nn.Sequential(*layers)