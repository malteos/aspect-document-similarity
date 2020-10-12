import collections
from typing import List

import numpy as np

from torch.optim.lr_scheduler import LambdaLR


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def flatten(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(dictionary, sep='__'):
    out_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = out_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return out_dict



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_categorical_one_hot_encoding_from_str(label_str, label_classes: List[str], label_sep=',', return_list=False):
    """
    Converts a single or list categorical labels into a one-hot-encoded vectors.
    (multi-label multi-class classification)

    good,bad => [1.0, 1.0]
    good => [1.0, 0.0]

    [good,bad], [good] => [ [1.0, 1.0], [1.0, 0.0] ]

    :param return_list:
    :param label_str:
    :param label_classes: Label classes
    :param label_sep: Label separator (default: ,)
    :return: np.array or List
    """
    if isinstance(label_str, List):
        # If input is a list of strings
        ls = [get_categorical_one_hot_encoding_from_str(ls, label_classes, label_sep, return_list) for ls in label_str]

        if return_list:
            return ls
        else:
            return np.array(ls)

    numerical_labels = [label_classes.index(l) for l in label_str.split(label_sep)]
    one_hot = np.zeros(len(label_classes))

    one_hot[numerical_labels] = 1.

    if return_list:
        return one_hot.tolist()
    else:
        return one_hot


def get_categorical_one_hot_encoding_from_str(label_str, label_classes: List[str], label_sep=',', return_list=False):
    """
    Converts a single or list categorical labels into a one-hot-encoded vectors.
    (multi-label multi-class classification)

    good,bad => [1.0, 1.0]
    good => [1.0, 0.0]

    [good,bad], [good] => [ [1.0, 1.0], [1.0, 0.0] ]

    :param return_list:
    :param label_str:
    :param label_classes: Label classes
    :param label_sep: Label separator (default: ,)
    :return: np.array or List
    """
    if isinstance(label_str, List):
        # If input is a list of strings
        ls = [get_categorical_one_hot_encoding(ls, label_classes, label_sep, return_list) for ls in label_str]

        if return_list:
            return ls
        else:
            return np.array(ls)

    numerical_labels = [label_classes.index(l) for l in label_str.split(label_sep)]
    one_hot = np.zeros(len(label_classes))

    one_hot[numerical_labels] = 1.

    if return_list:
        return one_hot.tolist()
    else:
        return one_hot


def highlight_max(data, color='green'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)