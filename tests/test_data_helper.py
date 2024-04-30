from unittest import TestCase


# TODO implement these tests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader

from sci.data_helpers import BaseSciDataHelper


class DataHelperTests(TestCase):
    def test_negative_sampling(self):
        raise NotImplementedError()

    def test_weighted_sampler(self):

        items_a = ['a'] * 10
        items_b = ['b'] * 3
        items_c = ['c'] * 5
        items = items_a + items_b + items_c

        dh = BaseSciDataHelper(label_col='label', labels=['a', 'b', 'c'], none_label=None)

        df = pd.DataFrame({'label': items})
        dh.set_label_encoder(df)

        label_weights, weights = dh.get_sampler_weights(df)

        ys = torch.tensor(dh.label_encoder.transform(items))

        sampler = WeightedRandomSampler(weights, num_samples=int(weights.sum()), replacement=True)

        dl = DataLoader(TensorDataset(ys), sampler=sampler, batch_size=4)

        out = []

        for batch in dl:
            yss = batch[0].numpy()
            out += dh.label_encoder.inverse_transform(yss).tolist()

        odf = pd.DataFrame({'label': out})

        print(df['label'].value_counts())
        print(odf['label'].value_counts())
