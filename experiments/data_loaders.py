import logging

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class XYDataLoader(DataLoader):
    """

    Batch consists only of X (item data) and Y (label)

    """
    def get_x_from_batch(self, batch):
        raise NotImplementedError()

    def get_y_from_batch(self, batch):
        raise NotImplementedError()


class DefaultXYDataLoader(XYDataLoader):
    """

    Last item of batch is Y, everything else is X.

    """
    def get_x_from_batch(self, batch):
        return batch[:-1]

    def get_y_from_batch(self, batch):
        return batch[-1]